#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w EMA filter and volume confirmation
# - Long when price > KAMA(14) AND 1w EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when price < KAMA(14) AND 1w EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit on opposite KAMA cross or when 1w EMA flips
# - Uses weekly EMA50 for strong trend filter to avoid whipsaws
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)

name = "1d_1w_kama_ema_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute KAMA(14) on 1d close
    close = prices['close'].values
    direction = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when price > KAMA AND 1w uptrend with volume spike
            if (close[i] > kama[i] and 
                close[i] > ema50_1w_aligned[i] and  # price above 1w EMA50
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price < KAMA AND 1w downtrend with volume spike
            elif (close[i] < kama[i] and 
                  close[i] < ema50_1w_aligned[i] and  # price below 1w EMA50
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses KAMA in opposite direction OR 1w EMA flips
            exit_signal = False
            if position == 1:  # Long position
                if close[i] < kama[i] or close[i] < ema50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if close[i] > kama[i] or close[i] > ema50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals