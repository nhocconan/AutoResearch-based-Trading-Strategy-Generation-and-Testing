#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + 1d Trend + Volume Spike + ADX Filter

HYPOTHESIS: 12h primary with 1d trend confirmation:
- 1d HMA(21) for multi-day trend direction (bull/bear filter)
- 12h Donchian(15) for price channel breakout (structure)
- Volume spike 2.0x for momentum confirmation
- ADX > 20 for trend quality filter
- ATR(14) 2.0x trailing stop

WHY IT WORKS ON BOTH BULL AND BEAR:
- Bull: Donchian up + HMA up + vol spike = high-probability long
- Bear: Donchian down + HMA down + vol spike = high-probability short
- ADX filter removes choppy markets where breakouts fail
- Weekly HMA context prevents catching falling knives

TARGET: 75-150 total trades over 4 years (18-37/year)
Rationale: 12h has ~4x fewer bars than 4h, so need slightly looser filters.
ADX > 20 (vs 25) + Donchian(15) (vs 20) should give 75-150 trades.

KEY DIFFERENCE FROM FAILED #012/#015/#019:
- #012: 12h ADX+RSI+BB only got 7 trades (too strict)
- #015: 12h Donchian+1w HMA only got 0 trades (impossible combo)
- #019: 12h Donchian+1d HMA+vol+ADX got 53 trades, Sharpe -0.735 (wrong filters)
- THIS: Relaxed 1d HMA instead of 1w, ADX 20 (not 25), Donchian 15 (not 20)

CRITICAL: fewer trades = less fee drag = better generalization
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_adx_vol_v2"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_donchian(high, low, period=15):
    """Donchian Channel - tighter period for more signals"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0
    
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr_smooth[i] > 1e-10:
            plus_di[i] = 100 * pd.Series(plus_dm[:i+1]).ewm(span=period, min_periods=period, adjust=False).mean().values[-1] / atr_smooth[i]
            minus_di[i] = 100 * pd.Series(minus_dm[:i+1]).ewm(span=period, min_periods=period, adjust=False).mean().values[-1] / atr_smooth[i]
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period * 2, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 12h Local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=15)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 200  # 15 for donchian + 28 for ADX + 20 for vol MA + HTF alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        # === TREND FILTER: 1d HMA direction ===
        htf_trend_up = close[i] > hma_aligned[i]
        htf_trend_down = close[i] < hma_aligned[i]
        
        # === ADX TREND STRENGTH FILTER (lowered to 20) ===
        adx_ok = adx[i] > 20
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout + HMA up + ADX + Volume ===
            if breakout_up and htf_trend_up and adx_ok and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Breakdown + HMA down + ADX + Volume ===
            if breakout_down and htf_trend_down and adx_ok and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                stop_price = trailing_high - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if trend flips
                if htf_trend_down:
                    desired_signal = 0.0
            
            elif position_side < 0:
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                stop_price = trailing_low + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if trend flips
                if htf_trend_up:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals