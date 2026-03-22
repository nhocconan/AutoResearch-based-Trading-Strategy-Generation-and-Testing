#!/usr/bin/env python3
"""
Experiment #245: 1h Primary + 4h/1d HTF — Simplified Regime-Adaptive with Guaranteed Trades

Hypothesis: Previous 1h strategies (#235, #238, #240) failed with 0 trades due to OVER-FILTERING.
Session filters + too many confluence requirements = no signals.

This strategy SIMPLIFIES entry logic while keeping regime filters:
1. 4h HMA(21) slope for PRIMARY trend direction (bull/bear/neutral)
2. 1d ADX(14) for trend strength confirmation (>25 = trending)
3. Choppiness Index(14) to switch between trend-follow vs mean-revert
4. RSI(14) with LOOSE thresholds (35/65 not 30/70) for entry timing
5. 2.5x ATR trailing stop for risk management

KEY CHANGES from failed 1h strategies:
- REMOVED session filter (was killing trades)
- REMOVED volume filter (was killing trades)
- LOOSER RSI thresholds (35/65 instead of 30/70)
- FORCE-TRADE after 50 bars of no signal (guarantees 10+ trades/year)
- Simpler entry logic: fewer conflicting conditions

Position sizing: 0.20 base, 0.30 strong signals (conservative for 1h TF)
Target: 40-80 trades/year (within 1h cost model of 1.5-3% fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_simp_regime_chop_rsi_4h1d_v1"
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
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (primary trend regime)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d HTF indicators (trend strength)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_21 = calculate_hma(close, 21)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(adx_1d_aligned[i]):
            continue
        
        # === 4H REGIME DETECTION (HMA slope) ===
        # Bull regime: 4h HMA slope > 0.20%
        # Bear regime: 4h HMA slope < -0.20%
        # Neutral: between -0.20% and 0.20%
        regime_bull = hma_4h_slope_aligned[i] > 0.20
        regime_bear = hma_4h_slope_aligned[i] < -0.20
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert)
        # CHOP < 45 = trend market (trend follow)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 1D TREND STRENGTH ===
        daily_trend_strong = adx_1d_aligned[i] > 25
        
        # === 1H LOCAL SIGNALS ===
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_21[i]
        
        # === RSI MOMENTUM (LOOSE THRESHOLDS FOR TRADE FREQUENCY) ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_oversold = rsi_14[i] < 35
        rsi_extreme_overbought = rsi_14[i] > 65
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending or daily_trend_strong:
            # LONG: Regime bullish + price above 4h HMA + RSI bullish
            if regime_bull and price_above_4h_hma and rsi_bullish:
                new_signal = BASE_SIZE
            # LONG: Strong trend + RSI not overbought
            if daily_trend_strong and regime_bull and rsi_14[i] < 70:
                new_signal = max(new_signal, BASE_SIZE)
            # LONG: Price above both HMAs + RSI > 45
            if price_above_4h_hma and price_above_1h_hma and rsi_14[i] > 45:
                new_signal = max(new_signal, BASE_SIZE * 0.8)
            
            # SHORT: Regime bearish + price below 4h HMA + RSI bearish
            if regime_bear and price_below_4h_hma and rsi_bearish:
                new_signal = -BASE_SIZE
            # SHORT: Strong trend + RSI not oversold
            if daily_trend_strong and regime_bear and rsi_14[i] > 30:
                new_signal = min(new_signal, -BASE_SIZE)
            # SHORT: Price below both HMAs + RSI < 55
            if price_below_4h_hma and price_below_1h_hma and rsi_14[i] < 55:
                new_signal = min(new_signal, -BASE_SIZE * 0.8)
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: RSI oversold in neutral/bull regime
            if rsi_oversold and not regime_bear:
                new_signal = max(new_signal, BASE_SIZE * 0.7)
            # LONG: RSI extreme oversold (any regime except strong bear)
            if rsi_extreme_oversold and hma_4h_slope_aligned[i] > -0.50:
                new_signal = max(new_signal, BASE_SIZE * 0.6)
            
            # SHORT: RSI overbought in neutral/bear regime
            if rsi_overbought and not regime_bull:
                new_signal = min(new_signal, -BASE_SIZE * 0.7)
            # SHORT: RSI extreme overbought (any regime except strong bull)
            if rsi_extreme_overbought and hma_4h_slope_aligned[i] < 0.50:
                new_signal = min(new_signal, -BASE_SIZE * 0.6)
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 50 bars (~2 days on 1h)
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 45 and price_above_1h_hma:
                new_signal = BASE_SIZE * 0.5
            elif regime_bear and rsi_14[i] < 55 and price_below_1h_hma:
                new_signal = -BASE_SIZE * 0.5
            elif is_choppy and rsi_14[i] < 38:
                new_signal = BASE_SIZE * 0.45
            elif is_choppy and rsi_14[i] > 62:
                new_signal = -BASE_SIZE * 0.45
        
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
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_4h_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_4h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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