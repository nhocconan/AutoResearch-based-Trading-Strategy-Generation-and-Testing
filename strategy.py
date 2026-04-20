# 1d_1w_RSI_Momentum_Volume
# Hypothesis: Daily RSI momentum with weekly trend filter and volume confirmation
# Works in bull (trend-following) and bear (mean-reversion via RSI extremes)
# Weekly EMA10 determines regime: price above = bull (RSI 40-70 long), below = bear (RSI 30-60 short)
# Volume > 1.5x average confirms momentum
# Target: 15-25 trades/year to avoid fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Momentum_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA10 for trend regime
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Daily RSI(14)
    close_1d = prices['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Daily volume ratio (current vs 20-day average)
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        rsi_val = rsi_1d[i]
        ema_weekly = ema10_1w_aligned[i]
        close_val = close_1d[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_val) or np.isnan(ema_weekly) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bull regime: price above weekly EMA10 -> look for RSI momentum longs
            if close_val > ema_weekly:
                if 40 < rsi_val < 70 and vol_ratio_val > 1.5:
                    signals[i] = 0.25
                    position = 1
            # Bear regime: price below weekly EMA10 -> look for RSI momentum shorts
            else:
                if 30 < rsi_val < 60 and vol_ratio_val > 1.5:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend change
            if rsi_val > 75 or close_val < ema_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or trend change
            if rsi_val < 25 or close_val > ema_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals