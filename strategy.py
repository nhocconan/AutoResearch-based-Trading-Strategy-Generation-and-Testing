#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot long/short with 1w trend filter and volume confirmation
# - Primary signal: Price touches Camarilla H3 (short) or L3 (long) levels on 1d
# - Trend filter: 1w EMA(21) slope - price above/below EMA for trend alignment
# - Volume confirmation: 1d volume > 1.3x 20-period average volume
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 1d
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Camarilla pivots capture reversals; trend filter avoids counter-trend whipsaws

name = "1d_1w_camarilla_pivot_trend_volume_v1"
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
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slope = np.diff(ema_21, prepend=ema_21[0])  # slope of EMA
    ema_trend_up = ema_slope > 0
    ema_trend_down = ema_slope < 0
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_up)
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_down)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Shift to use previous day's OHLC for today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4.0)
    l3 = pivot - (range_hl * 1.1 / 4.0)
    h4 = pivot + (range_hl * 1.1 / 2.0)
    l4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Pre-compute 1d volume confirmation
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    
    # Pre-compute 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # start after enough data for pivots
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches L4 (take profit) OR stoploss hit
            if close_1d[i] >= l4[i] or close_1d[i] < entry_price - 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches H4 (take profit) OR stoploss hit
            if close_1d[i] <= h4[i] or close_1d[i] > entry_price + 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touches with trend and volume filters
            if vol_spike[i]:
                # Long: price touches L3 AND uptrend on 1w
                if close_1d[i] <= l3[i] and ema_trend_up_aligned[i]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.25
                # Short: price touches H3 AND downtrend on 1w
                elif close_1d[i] >= h3[i] and ema_trend_down_aligned[i]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.25
    
    return signals