#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels + 1w trend filter + volume confirmation
# - Primary signal: price touches Camarilla H3 (short) or L3 (long) from 1d pivot calculation
# - Trend filter: 1w close > EMA(34) for longs, < EMA(34) for shorts (institutional weekly trend)
# - Volume filter: 1d volume > 1.3x 20-period average volume (momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(10) on 1d
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Pivot levels act as support/resistance; weekly trend filter avoids counter-trend trades

name = "1d_1w_camarilla_pivot_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Pre-compute 1d OHLC for Camarilla pivot calculation
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # We use shifted values to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First bar uses current values (no prior day)
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4.0)  # H3 = pivot + 1.1*(H-L)/4
    l3 = pivot - (range_hl * 1.1 / 4.0)  # L3 = pivot - 1.1*(H-L)/4
    h4 = pivot + (range_hl * 1.1 / 2.0)  # H4 = pivot + 1.1*(H-L)/2
    l4 = pivot - (range_hl * 1.1 / 2.0)  # L4 = pivot - 1.1*(H-L)/2
    
    # Pre-compute 1d volume spike filter
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    
    # Pre-compute 1d ATR(10) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(40, n):  # Start after warmup for 1w EMA
        # Skip if any required data is invalid
        if (np.isnan(ema_34_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(h4[i]) or np.isnan(l4[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR stoploss hit
            if close_1d[i] < l3[i] or close_1d[i] < entry_price - 1.5 * atr_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR stoploss hit
            if close_1d[i] > h3[i] or close_1d[i] > entry_price + 1.5 * atr_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touch with trend and volume filters
            if vol_spike[i]:
                # Long: price touches L3 from above in uptrend (close > EMA34)
                if low_1d[i] <= l3[i] and close_1d[i] > l3[i] and close_1d[i] > ema_34_aligned[i]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.25
                # Short: price touches H3 from below in downtrend (close < EMA34)
                elif high_1d[i] >= h3[i] and close_1d[i] < h3[i] and close_1d[i] < ema_34_aligned[i]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.25
    
    return signals