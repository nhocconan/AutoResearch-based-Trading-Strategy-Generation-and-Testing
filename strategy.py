#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H4 resistance in 1w uptrend (close > EMA50) with volume spike
# - Short when price breaks below Camarilla L4 support in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.30) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14)
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull (breakouts continue) and bear (mean reversion to pivot after extreme moves)

name = "1d_1w_camarilla_breakout_volume_trend_atr_v1"
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
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w ATR(14) for stoploss
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1w = np.zeros_like(tr)
    atr_14_1w[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1w[i] = (atr_14_1w[i-1] * (14-1) + tr[i]) / 14
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # 1w volume confirmation: > 2.0x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 1d Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5*(high-low)*1.1/2
    # L4 = close - 1.5*(high-low)*1.1/2
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d) * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss
            if close_1d[i] < entry_price - 2.5 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss
            if close_1d[i] > entry_price + 2.5 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Camarilla H4/L4 breakout with trend and volume filters
            if vol_spike_1w_aligned[i]:
                # Long signal: price breaks above H4 resistance in 1w uptrend
                if (close_1d[i] > camarilla_h4[i] and 
                    close_1d[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = close_1d[i]
                    entry_atr = atr_14_1w_aligned[i]
                    signals[i] = 0.30
                # Short signal: price breaks below L4 support in 1w downtrend
                elif (close_1d[i] < camarilla_l4[i] and 
                      close_1d[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = close_1d[i]
                    entry_atr = atr_14_1w_aligned[i]
                    signals[i] = -0.30
    
    return signals