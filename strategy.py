#!/usr/bin/env python3
"""
Experiment #185: 1h Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion with Session Filter

Hypothesis: Lower timeframe (1h) strategies fail due to either (1) too many trades causing fee drag,
or (2) too strict conditions causing 0 trades. This strategy balances both by:

1. 4H HMA(21) TREND BIAS: Direction filter (long only when 4h HMA sloping up, short when down)
2. 1H CHOPPINESS INDEX: Regime detection (CHOP>55 = range/mean-revert, CHOP<45 = trend/pullback)
3. 1H RSI(7) EXTREMES: Entry timing (RSI<25 long, RSI>75 short) — faster than RSI(14)
4. 1H BOLLINGER BANDS: Confirmation (price beyond 2.2std bands for extremes)
5. SESSION FILTER: Only trade 8-20 UTC (high liquidity, avoid Asian session noise)
6. VOLUME FILTER: Volume > 0.7x 20-bar average (confirm participation)
7. ATR VOLATILITY: Skip entries when ATR(14) > 2x ATR(50) (avoid panic/crash periods)

Why this should work on 1h:
- 4h trend filter prevents counter-trend trades (major failure mode)
- CHOP regime adapts to market conditions (range vs trend)
- Session filter reduces false signals during low liquidity
- Volume filter confirms real moves vs noise
- ATR filter avoids entering during extreme volatility (whipsaw risk)
- Multiple entry paths ensure we get 30-60 trades/year (not 0, not 200+)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for lower TF to reduce fee impact)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 30-60/year per symbol (strict enough to avoid fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_session_4h1d_v1"
timeframe = "1h"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
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

def extract_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    open_time_ms = prices["open_time"].values
    hours = ((open_time_ms / 1000 / 3600) % 24).astype(int)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.2)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume moving average
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract session hour
    session_hours = extract_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h TF)
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
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D TREND BIAS (stronger filter) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === VOLATILITY FILTER ===
        vol_normal = atr_14[i] < 2.0 * atr_50[i]  # Skip extreme vol periods
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= session_hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_sma_20[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 28
        rsi_overbought = rsi_7[i] > 72
        rsi_extreme_low = rsi_7[i] < 20
        rsi_extreme_high = rsi_7[i] > 80
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.6
        
        # Reduce size in extreme volatility
        if not vol_normal:
            current_size = current_size * 0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        long_confidence = 0
        
        # Path 1: Range market + RSI oversold + BB lower (mean revert)
        if is_range_market and rsi_oversold and price_below_bb_lower:
            long_score += 3
            long_confidence += 2
        
        # Path 2: 4h bullish trend + RSI pullback + price above 4h HMA
        if trend_4h_bullish and rsi_7[i] < 35 and price_above_4h_hma:
            long_score += 3
            long_confidence += 2
        
        # Path 3: 1d bullish + 4h pullback + RSI extreme
        if trend_1d_bullish and price_below_4h_hma and rsi_extreme_low:
            long_score += 4
            long_confidence += 3
        
        # Path 4: Simple oversold with volume confirmation (fallback for more trades)
        if rsi_7[i] < 25 and price_below_bb_lower and volume_ok:
            long_score += 2
            long_confidence += 1
        
        # Path 5: Range market + RSI extreme (aggressive mean revert)
        if is_range_market and rsi_extreme_low and in_session:
            long_score += 2
            long_confidence += 1
        
        # Apply filters to long entries
        if long_score >= 3:
            if in_session and vol_normal:
                new_signal = current_size
            elif in_session:
                new_signal = current_size * 0.6
        elif long_score >= 2 and bars_since_last_trade > 60:
            if in_session and vol_normal and volume_ok:
                new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        short_confidence = 0
        
        # Path 1: Range market + RSI overbought + BB upper
        if is_range_market and rsi_overbought and price_above_bb_upper:
            short_score += 3
            short_confidence += 2
        
        # Path 2: 4h bearish trend + RSI rally + price below 4h HMA
        if trend_4h_bearish and rsi_7[i] > 65 and price_below_4h_hma:
            short_score += 3
            short_confidence += 2
        
        # Path 3: 1d bearish + 4h rally + RSI extreme
        if trend_1d_bearish and price_above_4h_hma and rsi_extreme_high:
            short_score += 4
            short_confidence += 3
        
        # Path 4: Simple overbought with volume confirmation
        if rsi_7[i] > 75 and price_above_bb_upper and volume_ok:
            short_score += 2
            short_confidence += 1
        
        # Path 5: Range market + RSI extreme (aggressive mean revert)
        if is_range_market and rsi_extreme_high and in_session:
            short_score += 2
            short_confidence += 1
        
        # Apply filters to short entries
        if short_score >= 3:
            if in_session and vol_normal:
                new_signal = -current_size
            elif in_session:
                new_signal = -current_size * 0.6
        elif short_score >= 2 and bars_since_last_trade > 60:
            if in_session and vol_normal and volume_ok:
                new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~5 days on 1h) to ensure minimum trades
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if in_session and vol_normal:
                if trend_4h_bullish and rsi_7[i] < 35:
                    new_signal = current_size * 0.4
                elif trend_4h_bearish and rsi_7[i] > 65:
                    new_signal = -current_size * 0.4
                elif is_range_market and rsi_7[i] < 30:
                    new_signal = current_size * 0.35
                elif is_range_market and rsi_7[i] > 70:
                    new_signal = -current_size * 0.35
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and is_trend_market and trend_4h_bearish:
                regime_reversal = True
            if position_side < 0 and is_trend_market and trend_4h_bullish:
                regime_reversal = True
        
        # === RSI REVERSAL EXIT ===
        rsi_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_7[i] > 70:
                rsi_reversal = True
            if position_side < 0 and rsi_7[i] < 30:
                rsi_reversal = True
        
        if stoploss_triggered or regime_reversal or rsi_reversal:
            new_signal = 0.0
        
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