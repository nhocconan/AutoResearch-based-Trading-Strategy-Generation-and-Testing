#!/usr/bin/env python3
"""
1h Session Filter + 4h/1d Trend + Volume Spike (Experiment #86994)
Hypothesis: In BTC/ETH, 1h breakouts with volume confirmation during liquid 
sessions (08-20 UTC) aligned with 4h trend and 1d momentum filter capture 
sustainable moves. Uses discrete sizing (0.20) and tight entry conditions 
to target 15-37 trades/year. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 trend filter
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d RSI(14) momentum filter
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend and momentum filters
        bullish_bias = curr_close > ema_4h_aligned[i]
        bearish_bias = curr_close < ema_4h_aligned[i]
        bullish_momentum = rsi_1d_aligned[i] > 50
        bearish_momentum = rsi_1d_aligned[i] < 50
        
        if position == 0:
            # Look for entry signals
            # Long: price makes new high AND bullish bias AND bullish momentum AND volume spike
            long_entry = (curr_close > np.maximum.accumulate(close[:i])[-1]) and bullish_bias and bullish_momentum and vol_spike
            # Short: price makes new low AND bearish bias AND bearish momentum AND volume spike
            short_entry = (curr_close < np.minimum.accumulate(close[:i])[-1]) and bearish_bias and bearish_momentum and vol_spike
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: loss of bullish bias OR RSI overextended (>70) OR new low (mean reversion)
            if (curr_close < ema_4h_aligned[i]) or (rsi_1d_aligned[i] > 70) or (curr_close < np.minimum.accumulate(close[:i+1])[-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: loss of bearish bias OR RSI oversold (<30) OR new high (mean reversion)
            if (curr_close > ema_4h_aligned[i]) or (rsi_1d_aligned[i] < 30) or (curr_close > np.maximum.accumulate(close[:i+1])[-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_SessionFilter_4hEMA50_1dRSI_VolumeSpike"
timeframe = "1h"
leverage = 1.0