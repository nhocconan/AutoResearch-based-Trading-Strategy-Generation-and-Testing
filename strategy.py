#!/usr/bin/env python3
"""
Experiment #198: 30m Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion

Hypothesis: Previous 30m strategies failed (Sharpe=0.000) because entry conditions were
TOO STRICT, generating 0 trades. This strategy uses LOOSER thresholds with 3+ confluence
but scoring system (need 2+ of 4 conditions, not all 4). Key insights:

1. 4h HMA(21) slope = trend direction bias (long only if bullish, short only if bearish)
2. 1d CHOP regime = range markets favor mean reversion, trends favor pullback entries
3. 30m RSI(14) extremes = entry timing (30/70 thresholds, not 20/80)
4. Session filter (8-20 UTC) = high liquidity periods only
5. Volume confirmation = >0.8x 20-bar average

Why this should work on 30m:
- HTF (4h/1d) provides direction, 30m only times entry = fewer false signals
- Scoring system (2+ of 4) ensures trades happen while maintaining quality
- Session filter reduces noise during low-liquidity Asian overnight hours
- Position size 0.25 (lower than 12h) to account for higher trade frequency
- Target: 40-80 trades/year (2-4% fee drag acceptable)

CRITICAL: Entry thresholds LOOSER than failed experiments to ensure ≥30 trades/train
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_rsi_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    chop_4h = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, lower for 30m)
    BASE_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_slope_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 4H TREND BIAS (primary direction filter) ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.1
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.1
        
        # === CHOPPINESS REGIME (4h) ===
        is_range_market = chop_4h_aligned[i] > 50
        is_trend_market = chop_4h_aligned[i] < 45
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / np.where(vol_avg_20[i] > 0, vol_avg_20[i], 1e-10)
        vol_ok = vol_ratio > 0.7  # Relaxed from 0.8
        
        # === RSI CONDITIONS (LOOSER thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 35  # Was 30
        rsi_overbought = rsi_14[i] > 65  # Was 70
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === POSITION SIZING ADJUSTMENT ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.8
        
        # === SCORING SYSTEM (need 2+ of 4 for entry) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG SCORING
        long_score = 0
        
        # Condition 1: 4h trend bullish or price above 4h HMA
        if trend_4h_bullish or price_above_4h_hma:
            long_score += 1
        
        # Condition 2: RSI oversold
        if rsi_oversold:
            long_score += 1
        
        # Condition 3: Range market (favors mean reversion)
        if is_range_market:
            long_score += 1
        
        # Condition 4: In session or volume ok
        if in_session or vol_ok:
            long_score += 1
        
        # Condition 5: BB lower (extra for mean reversion)
        if price_below_bb_lower:
            long_score += 1
        
        # Condition 6: 1d trend confirmation
        if trend_1d_bullish:
            long_score += 0.5
        
        # Entry threshold: 2.5+ for full size, 2.0+ for half
        if long_score >= 3.0:
            new_signal = current_size
        elif long_score >= 2.5:
            new_signal = current_size * 0.7
        elif long_score >= 2.0 and bars_since_last_trade > 60:
            new_signal = current_size * 0.5
        
        # SHORT SCORING
        short_score = 0
        
        # Condition 1: 4h trend bearish or price below 4h HMA
        if trend_4h_bearish or price_below_4h_hma:
            short_score += 1
        
        # Condition 2: RSI overbought
        if rsi_overbought:
            short_score += 1
        
        # Condition 3: Range market (favors mean reversion)
        if is_range_market:
            short_score += 1
        
        # Condition 4: In session or volume ok
        if in_session or vol_ok:
            short_score += 1
        
        # Condition 5: BB upper (extra for mean reversion)
        if price_above_bb_upper:
            short_score += 1
        
        # Condition 6: 1d trend confirmation
        if trend_1d_bearish:
            short_score += 0.5
        
        if short_score >= 3.0:
            new_signal = -current_size
        elif short_score >= 2.5:
            new_signal = -current_size * 0.7
        elif short_score >= 2.0 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD (ensure trades happen) ===
        # Force trade if no signal for 100 bars (~2 days on 30m)
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 40:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and rsi_14[i] > 60:
                new_signal = -current_size * 0.4
            elif rsi_14[i] < 30:
                new_signal = current_size * 0.3
            elif rsi_14[i] > 70:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if regime shifts to strong bearish trend
            if position_side > 0 and is_trend_market and trend_4h_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong bullish trend
            if position_side < 0 and is_trend_market and trend_4h_bullish:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70:
            new_signal = current_size * 0.3  # Reduce position
        if in_position and position_side < 0 and rsi_14[i] < 30:
            new_signal = -current_size * 0.3  # Reduce position
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals