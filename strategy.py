#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h volume-weighted VWAP deviation with 4h trend filter and session filter
# Trades pullbacks to VWAP in trending markets (4h EMA21) during active hours (08-20 UTC).
# Uses volume-weighted average price (VWAP) as dynamic support/resistance.
# Long when price > VWAP and deviates below by >1.5% in uptrend; short when price < VWAP and deviates above by >1.5% in downtrend.
# Designed for low trade frequency (15-35/year) with high win rate in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA for trend filter (computed once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = typical_price * volume
    vwap_den = volume
    
    # Cumulative VWAP with reset each day
    vwap = np.full(n, np.nan)
    cum_num = 0.0
    cum_den = 0.0
    prev_day = None
    
    for i in range(n):
        day = prices.iloc[i]['open_time'].date()
        if prev_day is None or day != prev_day:
            cum_num = 0.0
            cum_den = 0.0
            prev_day = day
        cum_num += vwap_num[i]
        cum_den += vwap_den[i]
        if cum_den > 0:
            vwap[i] = cum_num / cum_den
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # 20% position size
    
    for i in range(21, n):  # Start after EMA warmup
        # Skip if outside session or missing data
        if not in_session[i] or np.isnan(vwap[i]) or np.isnan(ema_4h_aligned[i]):
            continue
        
        # Calculate deviation from VWAP as percentage
        if vwap[i] <= 0:
            continue
        dev_pct = (close[i] - vwap[i]) / vwap[i] * 100.0
        
        # Determine trend from 4h EMA slope (simple: current vs 3 periods ago)
        if i >= 3:
            ema_now = ema_4h_aligned[i]
            ema_prev = ema_4h_aligned[i-3]
            trending_up = ema_now > ema_prev
            trending_down = ema_now < ema_prev
        else:
            trending_up = False
            trending_down = False
        
        # Long: price below VWAP in uptrend (pullback to buy)
        if trending_up and dev_pct < -1.5 and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short: price above VWAP in downtrend (pullback to sell)
        elif trending_down and dev_pct > 1.5 and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: price crosses VWAP or trend change
        elif position == 1 and (close[i] >= vwap[i] or not trending_up):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= vwap[i] or not trending_down):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_VWAP_Pullback_Trend_Filter"
timeframe = "1h"
leverage = 1.0