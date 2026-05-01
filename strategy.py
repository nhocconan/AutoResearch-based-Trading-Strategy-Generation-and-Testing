#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume spike confirmation.
# Uses 4h EMA50 for trend direction and 1h volume > 2.0 * 20-period average for confirmation.
# Enters long when RSI < 30 in uptrend, short when RSI > 70 in downtrend.
# Exits when RSI crosses 50 (mean reversion completion).
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Discrete position sizing 0.20 to balance return and drawdown.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_RSI14_MeanReversion_4hEMA50_Trend_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1h volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(14, 20, 50) + 1  # 51 (for RSI14, volume MA20, and EMA50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_values[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_rsi = rsi_values[i]
        curr_volume = volume[i]
        
        # Trend filter: 4h EMA50 direction
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: RSI < 30 (oversold) AND uptrend AND volume confirmation
            if curr_rsi < 30 and uptrend and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) AND downtrend AND volume confirmation
            elif curr_rsi > 70 and downtrend and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on RSI crossing above 50 (mean reversion complete)
            if curr_rsi > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on RSI crossing below 50 (mean reversion complete)
            if curr_rsi < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals