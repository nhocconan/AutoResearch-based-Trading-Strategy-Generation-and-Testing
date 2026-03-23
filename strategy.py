#!/usr/bin/env python3
"""
Experiment #755: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + ADX Regime

Hypothesis: After 500+ failed strategies and analyzing why 1h strategies fail (#745, #750):
1. 1h strategies fail because they use 1h for trend direction (too noisy)
2. Solution: Use 4h HMA for trend BIAS, 1h Fisher Transform for ENTRY TIMING only
3. Fisher Transform (Ehlers) excels at catching reversals in bear/range markets (BTC 2025)
4. 1d ADX(14) filters trending vs ranging regimes (ADX>25 = trend, ADX<20 = range)
5. Looser Fisher thresholds (-1.2/+1.2 instead of -1.5/+1.5) ensure >=30 trades/train
6. ATR(14) trailing stop 2.5x protects against adverse moves
7. Session filter REMOVED — was killing trade frequency in prior 1h attempts

Strategy design:
1. 4h HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 1d ADX(14) for regime detection (trending vs ranging)
3. 1h Fisher Transform(9) for entry timing (crosses at extremes)
4. 1h ATR(14) for trailing stop
5. 1h Volume > 0.7x 20-period avg (looser than prior attempts)
6. Discrete signals: 0.0, ±0.25, ±0.30

Key improvements from failed 1h strategies (#745, #750):
- Removed session filter (was causing 0 trades)
- Looser Fisher thresholds for more entries
- 4h HMA for trend (not 1h indicators)
- Simpler volume filter (0.7x instead of 0.8x)
- Better hold logic to maintain positions

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_4h_adx_1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extremes. Long when Fisher crosses above -1.5.
    Short when Fisher crosses below +1.5.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, fisher_signal
    
    # Calculate typical price
    typical = (high + low + close) / 3
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = 0.0
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2 * (typical[i] - lowest) / price_range - 1
        
        # Apply exponential smoothing to normalized value
        if i == period:
            smoothed = normalized
        else:
            smoothed = 0.7 * normalized + 0.3 * (2 * (typical[i-1] - lowest) / price_range - 1)
        
        # Clamp to avoid division by zero
        smoothed = np.clip(smoothed, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + smoothed) / (1 - smoothed))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx, plus_di, minus_di
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR and DM using Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    fisher_1h, fisher_signal_1h = calculate_fisher_transform(high, low, close, period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    vol_avg_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    sma_50_1h = calculate_sma(close, period=50)
    sma_200_1h = calculate_sma(close, period=200)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    adx_1d_raw, plus_di_1d, minus_di_1d = calculate_adx(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crosses for entry timing
    prev_fisher = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(fisher_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        if np.isnan(sma_50_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        if np.isnan(vol_avg_1h[i]) or vol_avg_1h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (4h HTF HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1d ADX) ===
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_extreme_low = fisher_1h[i] < -1.2
        fisher_extreme_high = fisher_1h[i] > 1.2
        
        # Fisher cross detection
        fisher_cross_up = prev_fisher < -1.2 and fisher_1h[i] >= -1.2
        fisher_cross_down = prev_fisher > 1.2 and fisher_1h[i] <= 1.2
        
        # === VOLUME FILTER (looser than prior attempts) ===
        volume_ok = volume[i] > 0.7 * vol_avg_1h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50_1h[i]
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma50 = close[i] < sma_50_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (ADX > 25) ===
        if trending_regime:
            # Long: 4h bullish + Fisher cross up + volume + above SMA50
            if trend_4h_bullish and fisher_cross_up and volume_ok and above_sma50:
                desired_signal = BASE_SIZE
            
            # Short: 4h bearish + Fisher cross down + volume + below SMA50
            if trend_4h_bearish and fisher_cross_down and volume_ok and below_sma50:
                desired_signal = -BASE_SIZE
            
            # Trend continuation (no cross needed, just extreme Fisher)
            if trend_4h_bullish and fisher_extreme_low and volume_ok:
                desired_signal = REDUCED_SIZE
            
            if trend_4h_bearish and fisher_extreme_high and volume_ok:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME LOGIC (ADX < 20) ===
        elif ranging_regime:
            # Mean reversion long: Fisher extreme low + 4h bullish bias
            if fisher_extreme_low and trend_4h_bullish and volume_ok:
                desired_signal = REDUCED_SIZE
            
            # Mean reversion short: Fisher extreme high + 4h bearish bias
            if fisher_extreme_high and trend_4h_bearish and volume_ok:
                desired_signal = -REDUCED_SIZE
            
            # Pure mean reversion (no trend bias in strong range)
            if fisher_extreme_low and not trend_4h_bearish and volume_ok:
                desired_signal = REDUCED_SIZE
            
            if fisher_extreme_high and not trend_4h_bullish and volume_ok:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) ===
        else:
            # Conservative: only enter on Fisher crosses + trend alignment
            if fisher_cross_up and trend_4h_bullish and volume_ok and above_sma50:
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_down and trend_4h_bearish and volume_ok and below_sma50:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish and Fisher not extremely overbought
                if trend_4h_bullish and fisher_1h[i] < 1.5:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish and Fisher not extremely oversold
                if trend_4h_bearish and fisher_1h[i] > -1.5:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h turns bearish AND Fisher overbought
            if trend_4h_bearish and fisher_1h[i] > 1.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h turns bullish AND Fisher oversold
            if trend_4h_bullish and fisher_1h[i] < -1.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
        prev_fisher = fisher_1h[i]
    
    return signals