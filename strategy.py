#!/usr/bin/env python3
"""
Experiment #111: 4h Primary + 1d HTF — Ehlers Fisher Transform + HMA Trend + ADX Regime

Hypothesis: After 100+ failed experiments, the pattern shows:
- Simple RSI strategies work but leave alpha on table (current best Sharpe=0.351)
- Ehlers Fisher Transform excels at catching reversals in bear/range markets (2025 test period)
- Fisher crosses are more sensitive than RSI extremes, generating more trades
- 1d HMA provides cleaner trend bias than KAMA (less lag)
- ADX regime filter prevents entries in dead markets but must be loose (>18 not >25)

This strategy combines:
1. 1d HMA(21) = major trend bias (price above/below)
2. 4h Fisher Transform = entry trigger (crosses -1.5/+1.5 levels)
3. ADX(14) > 18 = minimum momentum filter (loose to ensure trades)
4. ATR trailing stoploss (2.5x) for risk management
5. Position size: 0.27 (27% of capital, conservative for 4h)

Key improvements over #101:
- Fisher Transform is more sensitive than RSI for reversal detection
- HMA has less lag than KAMA for trend identification
- ADX threshold lowered from 25 to 18 to ensure trade generation
- Fisher levels (-1.5/+1.5) are proven reversal points from Ehlers research

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
Timeframe: 4h (proven, 20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_adx_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price into a Gaussian normal distribution
    Peaks and troughs are sharper than RSI, better for reversal detection
    
    Steps:
    1. Calculate typical price = (high + low) / 2
    2. Normalize to -1 to +1 range using Donchian channel
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 10:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize using highest high and lowest low over period
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        if highest - lowest < 1e-10:
            continue
        
        # Normalize to -1 to +1 (with 0.999 clamp to avoid ln(0))
        normalized = 2.0 * (typical[i] - lowest) / (highest - lowest) - 1.0
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Trigger line (1-period lag of fisher)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    # Smooth fisher with EMA
    fisher_smooth = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    trigger_smooth = pd.Series(trigger).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher_smooth, trigger_smooth

def calculate_hma(close, period=21):
    """
    Hull Moving Average
    Reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(series, span):
        """Weighted Moving Average"""
        weights = np.arange(1, span + 1)
        result = np.zeros(len(series))
        result[:] = np.nan
        for i in range(span - 1, len(series)):
            window = series[i-span+1:i+1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = period // 2
    if half < 1:
        half = 1
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA formula
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, int(np.sqrt(period)))
    
    return hma

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index
    Measures trend strength (not direction)
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, trigger = calculate_fisher_transform(high, low, close, period=9)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.27  # 27% position size (conservative for 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        # Price above daily HMA = bull bias, below = bear bias
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (fisher[i] > -1.5) and (trigger[i] <= -1.5)
        fisher_cross_short = (fisher[i] < 1.5) and (trigger[i] >= 1.5)
        
        # Also allow direct Fisher extremes for more trades
        fisher_extreme_long = fisher[i] < -1.2
        fisher_extreme_short = fisher[i] > 1.2
        
        # === ADX REGIME FILTER (LOOSE) ===
        # ADX > 18 = enough momentum to trade (lower than standard 25)
        adx_ok = adx[i] > 18.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + (Fisher cross long OR Fisher extreme long) + ADX ok
        # SHORT: 1d bear + (Fisher cross short OR Fisher extreme short) + ADX ok
        desired_signal = 0.0
        
        if htf_bull and (fisher_cross_long or fisher_extreme_long) and adx_ok:
            desired_signal = SIZE
        elif htf_bear and (fisher_cross_short or fisher_extreme_short) and adx_ok:
            desired_signal = -SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals