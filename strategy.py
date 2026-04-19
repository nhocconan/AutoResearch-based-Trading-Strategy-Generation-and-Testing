# 1d_KAMA_RSI_Chop_v1 - 1d KAMA trend with RSI momentum and Choppiness filter
# Trend-following strategy optimized for low trade frequency (<20 trades/year)
# Uses KAMA for trend direction, RSI for momentum confirmation, Choppiness Index for regime filter
# Works in bull markets by catching trends and avoids choppy/ranging markets
# Only takes trades when market is trending (CHOP < 38.2) to reduce whipsaw

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) - Trend ===
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.sum, 'axis') else np.abs(np.diff(close)).sum()
    # More robust volatility calculation
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+er_period]))) if i+er_period <= len(close) else 0 
                          for i in range(len(close))])
    volatility[:er_period-1] = 0  # Not enough data
    
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.concatenate([np.full(er_period-1, np.nan), er])
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    sc = np.nan_to_num(sc, nan=0.0)
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_period-1] = close[er_period-1]  # Start with first available close
    for i in range(er_period, len(close)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (Relative Strength Index) - Momentum ===
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        for i in range(rsi_period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(rsi_period, np.nan), rsi[rsi_period:]])
    
    # === Choppiness Index - Regime Filter ===
    chop_period = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (using Wilder's smoothing)
    atr = np.full_like(close, np.nan)
    if len(tr) >= chop_period:
        atr[chop_period-1] = np.nanmean(tr[1:chop_period])  # Skip first NaN
        for i in range(chop_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * (chop_period-1) + tr[i]) / chop_period
    
    # Sum of ATR over period
    atr_sum = np.full_like(close, np.nan)
    for i in range(chop_period-1, len(close)):
        start_idx = i - chop_period + 1
        if start_idx >= 0:
            atr_sum[i] = np.nansum(atr[start_idx:i+1])
    
    # Max and min close over period
    max_high = np.full_like(close, np.nan)
    min_low = np.full_like(close, np.nan)
    for i in range(chop_period-1, len(close)):
        start_idx = i - chop_period + 1
        max_high[i] = np.nanmax(high[start_idx:i+1])
        min_low[i] = np.nanmin(low[start_idx:i+1])
    
    # Choppiness Index
    chop = np.full_like(close, np.nan)
    for i in range(chop_period-1, len(close)):
        if not np.isnan(atr_sum[i]) and atr_sum[i] > 0:
            range_val = max_high[i] - min_low[i]
            if range_val > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / range_val) / np.log10(chop_period)
    
    # === Weekly Trend Filter (HTF) ===
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) > 0:
        close_1w = df_1w['close'].values
        # Weekly EMA(34) for trend
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    else:
        ema_34_1w_aligned = np.full_like(close, np.nan)
    
    # === Volume Confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, er_period, rsi_period*2, chop_period)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Market regime: only trade when trending (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long entry: price above KAMA (uptrend) + RSI > 50 (momentum) + weekly uptrend + volume
            if (price > kama_val and rsi_val > 50 and 
                ema_34_1w_val > 0 and  # Weekly EMA has value (not NaN)
                price > ema_34_1w_val and  # Price above weekly EMA
                is_trending and 
                vol > 1.2 * vol_ma):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA (downtrend) + RSI < 50 (momentum) + weekly downtrend + volume
            elif (price < kama_val and rsi_val < 50 and 
                  ema_34_1w_val > 0 and
                  price < ema_34_1w_val and  # Price below weekly EMA
                  is_trending and 
                  vol > 1.2 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA OR RSI < 40 (loss of momentum) OR choppy market
            if (price < kama_val or rsi_val < 40 or chop_val >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA OR RSI > 60 (loss of momentum) OR choppy market
            if (price > kama_val or rsi_val > 60 or chop_val >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals