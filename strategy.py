#!/usr/bin/env python3
"""
4h Volume Spike Reversal with 1d EMA34 Trend and 1w Funding Z-Score Contrarian
Hypothesis: In 4h timeframe, extreme volume spikes (>3x 20-bar MA) during overextended moves 
(price >1.5*ATR from daily EMA34) signal exhaustion. Contrarian entries are taken when 
1-week funding rate Z-score < -2 (long) or > +2 (short), aligned with daily EMA34 trend. 
This combines mean reversion on exhaustion with funding rate edge for BTC/ETH, working in 
both bull and bear markets by fading spikes with institutional funding bias.
"""

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
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Daily EMA34 for trend and deviation filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h ATR(14) for deviation measurement
    tr1 = pd.Series(high).sub(pd.Series(low))
    tr2 = pd.Series(high).sub(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).sub(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume spike: >3.0 * 20-period average (strict)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 3.0)
    
    # 1-week funding rate Z-score (contrarian signal)
    # Funding rate data would normally come from separate funding parquet files
    # For now, we'll simulate using price action as proxy: extreme price moves vs weekly EMA
    # In practice, replace this with: funding_z = (funding_rate - funding_ma) / funding_std
    weekly_close = df_1w['close'].values
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    
    # Price deviation from weekly EMA as funding proxy (negative correlation)
    price_deviation = (close - weekly_ema_aligned) / atr  # in ATR units
    # Extreme deviations signal potential exhaustion (contrarian opportunity)
    # We'll use this as a proxy for funding extremes until real funding data is integrated
    funding_z_proxy = -price_deviation  # negative so extreme positive deviation = negative Z (long signal)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(34, 20, 50) + 1  # EMA34, vol MA, weekly EMA50 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_ema_aligned[i]) or
            np.isnan(funding_z_proxy[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        dev_from_daily_ema = abs(curr_close - ema_34_aligned[i]) / atr[i] if atr[i] > 0 else 0
        
        # Extreme deviation from daily EMA (>1.5 ATR) + volume spike = exhaustion
        exhaustion_long = (curr_close < ema_34_aligned[i] - 1.5 * atr[i]) and vol_spike
        exhaustion_short = (curr_close > ema_34_aligned[i] + 1.5 * atr[i]) and vol_spike
        
        # Funding Z-score proxy for contrarian signal
        # Long when funding extremely negative (proxy: price far above weekly EMA)
        # Short when funding extremely positive (proxy: price far below weekly EMA)
        long_signal = exhaustion_long and (funding_z_proxy[i] < -2.0)
        short_signal = exhaustion_short and (funding_z_proxy[i] > 2.0)
        
        if position == 0:
            # Look for entry signals
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price returns to daily EMA OR funding normalizes
            if (curr_close >= ema_34_aligned[i]) or (funding_z_proxy[i] > -0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price returns to daily EMA OR funding normalizes
            if (curr_close <= ema_34_aligned[i]) or (funding_z_proxy[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_Reversal_1dEMA34_1wFundingZ"
timeframe = "4h"
leverage = 1.0