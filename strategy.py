#!/usr/bin/env python3
"""
Experiment #007: 6h Camarilla Pivots + 1d Trend + Volume Confirmation

HYPOTHESIS: Camarilla pivot levels provide precise S/R with proven edge.
- R3/S3: fade levels (mean reversion)
- R4/S4: breakout continuation levels
- 1d trend filter ensures we fade UP in bull markets, short DOWN in bear
- Volume spike confirms institutional flow
- 6h = sweet spot: enough signals for 75-200 trades, not as noisy as 4h

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull (2021, 2024): Price dips to S3/S4, bounces with volume = strong longs
- Bear (2022): Rallies to R3/R4, rejected with volume = strong shorts
- Range: R3/S3 fades work perfectly in chop

TARGET: 75-200 total trades over 4 years (19-50/year). HARD MAX: 300.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivots_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla_pivots(high, low, close, period=1):
    """
    Camarilla Pivot Levels
    R4 = close + (high - low) * 1.1
    R3 = close + (high - low) * 1.1/2
    R2 = close + (high - low) * 1.1/4
    R1 = close + (high - low) * 1.1/8
    S1 = close - (high - low) * 1.1/8
    S2 = close - (high - low) * 1.1/4
    S3 = close - (high - low) * 1.1/2
    S4 = close - (high - low) * 1.1
    """
    n = len(close)
    r4 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(period, n):
        h = high[i]
        l = low[i]
        c = close[i]
        rng = h - l
        
        r4[i] = c + rng * 1.1
        r3[i] = c + rng * 1.1 / 2.0
        s3[i] = c - rng * 1.1 / 2.0
        s4[i] = c - rng * 1.1
    
    return r4, r3, s3, s4

def calculate_adx(high, low, close, period=14):
    """ADX - Average Directional Index (trend strength)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX = smoothed DX
    adx = np.full(n, np.nan)
    adx[period] = np.mean(dx[period:period+period])
    
    for i in range(period + 1, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

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

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA(50) for trend - bullish when price > SMA
    sma_50_1d = calculate_sma(df_1d['close'].values, period=50)
    sma_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 1d Camarilla for HTF levels
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla_pivots(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=1
    )
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # 6h Camarilla pivots
    r4_6h, r3_6h, s3_6h, s4_6h = calculate_camarilla_pivots(high, low, close, period=1)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # 6h SMA(50) for local trend
    sma_50_local = calculate_sma(close, period=50)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # SMA50 + ATR14 + volume MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d) ===
        trend_up_1d = close[i] > sma_aligned[i]
        trend_down_1d = close[i] < sma_aligned[i]
        
        # === LOCAL TREND (6h) ===
        local_trend_up = close[i] > sma_50_local[i]
        local_trend_down = close[i] < sma_50_local[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === ADX: allow entries when trend is not too weak ===
        strong_trend = adx[i] > 18  # ADX below 18 = weak/no trend
        
        # === Camarilla Proximity (price near key levels) ===
        # Tolerance: within 0.5% of level
        tolerance = close[i] * 0.005
        
        at_s3 = abs(close[i] - s3_6h[i]) < tolerance
        at_s4 = abs(close[i] - s4_6h[i]) < tolerance
        at_r3 = abs(close[i] - r3_6h[i]) < tolerance
        at_r4 = abs(close[i] - r4_6h[i]) < tolerance
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price at S3/S4 support + uptrend + volume ===
            # S4 breakout continuation (stronger)
            # S3 bounce (mean reversion)
            
            if trend_up_1d and (at_s4 or at_s3):
                if vol_spike and strong_trend:
                    desired_signal = SIZE
                    if at_s4:
                        # S4 breakout = stronger signal
                        pass  # same size, but S4 is higher quality
            
            # === SHORT: Price at R3/R4 resistance + downtrend + volume ===
            if trend_down_1d and (at_r4 or at_r3):
                if vol_spike and strong_trend:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5x ATR) ===
        if in_position:
            if position_side > 0:
                # Stop if price falls 2.5 ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if 1d trend flips
                if trend_down_1d:
                    desired_signal = 0.0
                
                # Take profit at R3 (aggressive) if we're at S3
                if at_r3 and position_side > 0:
                    # Close long near resistance
                    if close[i] > r3_6h[i] * 0.998:
                        desired_signal = 0.0
            
            elif position_side < 0:
                # Stop if price rises 2.5 ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if 1d trend flips
                if trend_up_1d:
                    desired_signal = 0.0
                
                # Take profit at S3 (aggressive) if we're at R3
                if at_s3 and position_side < 0:
                    if close[i] < s3_6h[i] * 1.002:
                        desired_signal = 0.0
        
        # === MINIMUM HOLD: 6 bars (36h) to avoid fee churn ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals