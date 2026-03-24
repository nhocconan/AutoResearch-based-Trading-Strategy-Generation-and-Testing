#!/usr/bin/env python3
"""
Experiment #403: 6h Primary + 1d/1w HTF — Fisher Transform + ADX Regime v1

Hypothesis: Previous 6h strategies failed because RSI is too slow for reversal
detection in bear/range markets. Fisher Transform (Ehlers) provides faster
reversal signals with better timing. Combined with ADX regime filter and
asymmetric HTF bias, this should improve entry timing while maintaining
trade frequency.

Key innovations from failed experiments:
1. Fisher Transform (period=9) instead of RSI - catches reversals faster
2. Asymmetric entries: only long when 1w HTF bull, only short when 1w HTF bear
3. ADX hysteresis for regime stability (enter 25, exit 18)
4. Volume confirmation stricter (1.5x SMA not 1.2x)
5. Donchian breakout confirmation on trend entries

Regime Detection:
- ADX > 25 = trending → Fisher breakout entries with Donchian confirm
- ADX < 20 = choppy → Fisher mean reversion at extremes
- ADX 20-25 = use previous regime (hysteresis)

Entry Logic:
- Trending Long: Fisher > -1.5 (cross up) + 1w HMA bull + Donchian breakout
- Trending Short: Fisher < +1.5 (cross down) + 1w HMA bear + Donchian breakdown
- Choppy Long: Fisher < -1.8 + 1d HMA bull (oversold bounce)
- Choppy Short: Fisher > +1.8 + 1d HMA bear (overbought fade)

Position sizing: 0.25 base, 0.30 when 1w HTF aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better reversal detection than RSI in bear/range markets
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest)
    3. Transform: 0.5 * ln((1 + x) / (1 - x))
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over lookback
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = (typical[i] - lowest) / price_range
        
        # Clamp to avoid division by zero in log
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    return fisher

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, period=9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis (enter 25, exit 18)
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with ADX (hysteresis) ===
        # Trending: ADX > 25
        # Choppy: ADX < 20
        # Otherwise: use previous regime
        
        is_trending = adx[i] > 25.0
        is_choppy = adx[i] < 20.0
        
        if is_trending:
            current_regime = 1
        elif is_choppy:
            current_regime = 2
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS (1w for primary direction, 1d for confirmation) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 = long signal
        # Fisher crosses below +1.5 = short signal
        fisher_cross_long = False
        fisher_cross_short = False
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > +1.8
        
        if i > 0 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            # Cross above -1.5 (from below)
            if fisher[i-1] < -1.5 and fisher[i] >= -1.5:
                fisher_cross_long = True
            # Cross below +1.5 (from above)
            if fisher[i-1] > +1.5 and fisher[i] <= +1.5:
                fisher_cross_short = True
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === VOLUME CONFIRMATION (stricter: 1.5x not 1.2x) ===
        vol_confirm = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # === ENTRY LOGIC (ASYMMETRIC based on 1w HTF) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (breakout + trend alignment)
        if current_regime == 1:
            # Long: Fisher cross + 1w bull + (breakout OR HMA bull)
            if htf_1w_bull and fisher_cross_long:
                if breakout_long or hma_bull:
                    desired_signal = SIZE_STRONG if vol_confirm else SIZE_BASE
            
            # Short: Fisher cross + 1w bear + (breakdown OR HMA bear)
            elif htf_1w_bear and fisher_cross_short:
                if breakout_short or hma_bear:
                    desired_signal = -SIZE_STRONG if vol_confirm else -SIZE_BASE
        
        # REGIME 2: CHOPPY (Fisher mean reversion - ASYMMETRIC)
        elif current_regime == 2:
            # Long: Fisher oversold + 1d bull + above SMA200
            if fisher_oversold and htf_1d_bull and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: Fisher overbought + 1d bear + below SMA200
            elif fisher_overbought and htf_1d_bear and below_sma200:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals