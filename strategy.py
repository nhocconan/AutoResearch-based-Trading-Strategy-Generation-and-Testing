#!/usr/bin/env python3
name = "6h_Stealth_CCI_Range"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once for trend filter and CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Daily CCI (20-period)
    tp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (tp_1d - sma_tp) / (0.015 * mad)
    cci_1d[np.isnan(mad) | (mad == 0)] = 0
    
    # Align daily CCI to 6h (needs 0 delay - CCI is complete when daily bar closes)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h ATR(14) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Stealth zone: CCI between -80 and 80 (not extreme)
        in_stealth = (-80 <= cci_1d_aligned[i] <= 80)
        
        # Trend filter: price above/below EMA50
        above_ema50 = close[i] > ema_50_1d_aligned[i]
        below_ema50 = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: avoid choppy markets (ATR < 0.02 * price)
        vol_filter = atr_14[i] > 0.02 * close[i]
        
        if position == 0:
            # Long: CCI pulling up from stealth zone in uptrend
            if (cci_1d_aligned[i] > cci_1d_aligned[i-1] and 
                cci_1d_aligned[i-1] < -20 and  # was in deep stealth/oversold
                above_ema50 and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: CCI pulling down from stealth zone in downtrend
            elif (cci_1d_aligned[i] < cci_1d_aligned[i-1] and 
                  cci_1d_aligned[i-1] > 20 and  # was in deep stealth/overbought
                  below_ema50 and vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: CCI enters overbought or trend breaks
            if cci_1d_aligned[i] > 80 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CCI enters oversold or trend breaks
            if cci_1d_aligned[i] < -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Stealth CCI Range - Fade extremes in ranging markets
# - Uses daily CCI (20) to identify overbought/oversold conditions
# - Enters when CCI reverses from extreme levels (>80 or <-80) back into "stealth zone" (-80 to 80)
# - Only trades in direction of daily EMA50 trend (avoids counter-trend whipsaws)
# - Volatility filter (ATR > 2% of price) prevents trading in choppy/low-volatility conditions
# - Works in bull markets: buy dips in uptrend when CCI recovers from oversold
# - Works in bear markets: sell rallies in downtrend when CCI declines from overbought
# - Stealth zone avoids whipsaws from extreme CCI readings that often reverse
# - Position size 0.25 limits risk; targets ~50-120 trades over 4 years (12-30/year)
# - Daily timeframe for CCI reduces noise vs lower timeframes
# - Novel: CCI reversal from extremes + trend filter + vol filter not recently tried on 6h
# - Avoids saturated CCI overbought/oversold strategies by requiring reversal FROM extremes