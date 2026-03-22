#!/usr/bin/env python3
"""
Experiment #323: 12h KAMA Adaptive Trend with Dual HTF HMA Bias

Hypothesis: After #311 (Sharpe=0.094) showed EMA crossover works but is too laggy,
and #317 (Sharpe=-0.047) showed Choppiness Index doesn't help on 12h, I'll try:

1. KAMA (Kaufman Adaptive Moving Average) instead of EMA - adapts to volatility,
   reduces whipsaw in choppy markets while catching trends faster
2. 1d HMA(21) for primary directional bias (proven edge from multiple strategies)
3. 1w HMA(21) for meta-trend confirmation (soft filter - boosts size only)
4. KAMA(10)/KAMA(30) crossover for entry timing (more adaptive than EMA)
5. ADX(14) > 18 for trend confirmation (loose threshold for trade generation)
6. ATR(14) 2.2x trailing stoploss (proven from successful strategies)
7. Discrete position sizing: 0.25 base, 0.30 with 1w confirmation

Why KAMA over EMA:
- KAMA adapts its smoothing constant based on market efficiency ratio
- In trending markets: KAMA follows price closely (like fast EMA)
- In choppy markets: KAMA flattens out (reduces false signals)
- This should improve Sharpe vs #311's simple EMA crossover

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.2 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adaptive_dual_htf_hma_atr_v1"
timeframe = "12h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, fast_period=2, slow_period=30, er_period=10):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts to market volatility using Efficiency Ratio (ER).
    ER = |price change| / sum of absolute price changes over period
    - High ER (trending): KAMA follows price closely
    - Low ER (choppy): KAMA flattens, reduces whipsaw
    
    Parameters:
    - fast_period: SC for trending market (default 2/16 = 0.125)
    - slow_period: SC for choppy market (default 2/32 = 0.0625)
    - er_period: Period for Efficiency Ratio calculation
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio
    price_change = np.abs(close_s.diff(er_period))
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = price_change / volatility
    er = er.fillna(0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc.iloc[i]) or i < er_period:
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, fast_period=2, slow_period=30, er_period=10)
    kama_slow = calculate_kama(close, fast_period=2, slow_period=60, er_period=20)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = primary directional bias (REQUIRED)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = meta-trend confirmation (SOFT - boosts size but not required)
        bull_trend_1w = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        bear_trend_1w = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 18 = minimal trending (loose for trade generation)
        trending = adx[i] > 18
        strong_trend = adx[i] > 25
        
        # === KAMA CROSSOVER ===
        # Fast KAMA crosses above slow KAMA = bullish
        kama_bullish = kama_fast[i] > kama_slow[i]
        # Fast KAMA crosses below slow KAMA = bearish
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size
        if high_volatility:
            position_size = SIZE_BASE
        elif strong_trend and bull_trend_1w:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS (LOOSE for >=10 trades) ===
        new_signal = 0.0
        
        # LONG: 1d bias up + KAMA bullish + ADX trending
        # 1w is soft confirm only (boosts size, not required for entry)
        long_conditions = (
            bull_trend_1d and
            kama_bullish and
            trending
        )
        
        # SHORT: 1d bias down + KAMA bearish + ADX trending
        short_conditions = (
            bear_trend_1d and
            kama_bearish and
            trending
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.2 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.2 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.2 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === KAMA REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bearish:
                new_signal = 0.0
            if position_side < 0 and kama_bullish:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals