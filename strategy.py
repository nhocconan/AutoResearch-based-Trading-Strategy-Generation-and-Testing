#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-period daily Donchian high with weekly uptrend (weekly close > weekly EMA50) and daily volume spike
# - Short when price breaks below 20-period daily Donchian low with weekly downtrend (weekly close < weekly EMA50) and daily volume spike
# - Uses 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
# - Daily volume > 2.0x 20-period average confirms breakout strength
# - Weekly EMA50 filter ensures trading with weekly trend direction (more robust than daily)
# - Discrete position sizing (0.30) to balance return and drawdown
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) (tighter stop for better risk control)

name = "1d_1w_donchian_volume_trend_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute weekly indicators
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute daily indicators
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # Daily Donchian(20) channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume confirmation: > 2.0x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * avg_volume_20)
    
    # Daily ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = np.zeros_like(tr)
    atr_14[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * (14-1) + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss
            if close_1d[i] < entry_price - 2.0 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss
            if close_1d[i] > entry_price + 2.0 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long signal: price breaks above Donchian high in weekly uptrend
                if (close_1d[i] > donchian_high[i] and 
                    close_1d[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = close_1d[i]
                    entry_atr = atr_14[i]
                    signals[i] = 0.30
                # Short signal: price breaks below Donchian low in weekly downtrend
                elif (close_1d[i] < donchian_low[i] and 
                      close_1d[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = close_1d[i]
                    entry_atr = atr_14[i]
                    signals[i] = -0.30
    
    return signals