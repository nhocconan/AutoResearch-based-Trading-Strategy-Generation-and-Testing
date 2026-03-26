#!/usr/bin/env python3
"""
Experiment #011: 6h Elder Ray + Volume Spike + ADX Regime

HYPOTHESIS: Elder Ray (High - EMA(13), Low - EMA(13)) measures institutional 
pressure beyond the EMA. Bull Power > 0 confirms bulls can push beyond EMA.
Bear Power < 0 confirms bears pushing below EMA. This is distinct from RSI/MACD
because it measures WHERE price closes relative to EMA extremes, not momentum.

WHY NOVEL: Elder Ray hasn't appeared in any of the 27 failed attempts.
DB reference: Elder Ray is TIER 4 in program.md, untested in experiments.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Bull Power positive confirms buyers aggressive
- Bear markets: Bear Power negative confirms sellers aggressive  
- Range: ADX < 20 prevents trading in chop
- Volume spike confirms institutional involvement at key levels

TARGET: 75-150 total trades over 4 years (19-37/year).
SIZE: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_vol_adx_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Plus and Minus DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth
    smooth_plus = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    smooth_minus = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    smooth_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate DI
    di_plus = np.zeros(n, dtype=np.float64)
    di_minus = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if smooth_tr[i] > 0:
            di_plus[i] = 100.0 * smooth_plus[i] / smooth_tr[i]
            di_minus[i] = 100.0 * smooth_minus[i] / smooth_tr[i]
    
    # DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    return adx

def calculate_elder_ray(high, low, close, ema_period=13):
    """
    Elder Ray by Alexander Elder
    Bull Power = High - EMA(close, period)
    Bear Power = Low - EMA(close, period)
    
    Interpretation:
    - Bull Power > 0: Bulls can push above EMA
    - Bear Power < 0: Bears can push below EMA
    """
    n = len(close)
    ema = calculate_ema(close, ema_period)
    
    bull_power = np.zeros(n, dtype=np.float64)
    bear_power = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if not np.isnan(ema[i]):
            bull_power[i] = high[i] - ema[i]
            bear_power[i] = low[i] - ema[i]
    
    return bull_power, bear_power, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for multi-timeframe trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend direction
    ema_1d = calculate_ema(df_1d['close'].values, span=21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d ADX for regime
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Elder Ray
    bull_power, bear_power, ema_6h = calculate_elder_ray(high, low, close, ema_period=13)
    
    # Calculate 6h ADX
    adx_6h = calculate_adx(high, low, close, period=14)
    
    # ATR for stops
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    # Warmup
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(bull_power[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_6h[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (ADX < 25 = ranging, allow mean reversion) ===
        adx_trending = adx_6h[i] > 25
        
        # === 1d TREND BIAS ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i] if not np.isnan(ema_1d_aligned[i]) else True
        adx_1d_trending = adx_1d_aligned[i] > 20 if not np.isnan(adx_1d_aligned[i]) else False
        
        # === ELDER RAY SIGNALS ===
        bull = bull_power[i]
        bear = bear_power[i]
        ema = ema_6h[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        # Long: Bull Power > 0 + price above 6h EMA + volume confirmation
        if bull > 0 and close[i] > ema:
            if vol_spike or adx_trending:  # Either volume or trend confirmation
                desired_signal = SIZE
        
        # Short: Bear Power < 0 + price below 6h EMA + volume confirmation
        if bear < 0 and close[i] < ema:
            if vol_spike or adx_trending:
                desired_signal = -SIZE
        
        # === ADDITIONAL FILTER: TREND CONFLICT ===
        # In strong 1d uptrend (price >> 1d EMA), only allow longs
        if price_above_1d_ema and adx_1d_trending:
            if desired_signal < 0:  # Cancel shorts in strong uptrend
                desired_signal = 0.0
        
        # In strong 1d downtrend (price << 1d EMA), only allow shorts
        if not price_above_1d_ema and adx_1d_trending:
            if desired_signal > 0:  # Cancel longs in strong downtrend
                desired_signal = 0.0
        
        # === STOPLOSS CHECK ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT at 2R ===
        if in_position and position_side > 0:
            profit = close[i] - entry_price
            if profit >= 2.0 * entry_atr:
                desired_signal = SIZE / 2  # Half position at 2R
        
        if in_position and position_side < 0:
            profit = entry_price - close[i]
            if profit >= 2.0 * entry_atr:
                desired_signal = -SIZE / 2
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals