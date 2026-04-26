#!/usr/bin/env python3
"""
1h_VolumeSpike_Reversal_HTFTrend_v1
Hypothesis: Mean reversion on volume spikes aligned with 4h/1d trend.
- Long: 4h/1d uptrend + 1h volume spike (>2.0x 20MA) + RSI(14) < 30 (oversold)
- Short: 4h/1d downtrend + 1h volume spike (>2.0x 20MA) + RSI(14) > 70 (overbought)
- Exit: RSI crosses back to neutral (40-60 range) or 1h close beyond 1.5*ATR from spike candle
- Session filter: 08-20 UTC only
- Discrete sizing: 0.20 to limit fee churn
- Target: 60-120 trades over 4 years (15-30/year) - avoids fee drag while capturing reversals
- Works in bull/bear: uses HTF trend for direction, volume spike for exhaustion signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex from parquet
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Get 1d data for stronger trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 4h for trend
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate EMA(34) on 1d for stronger trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility and stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI(14) for mean reversion signals
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])  # prepend 0 for first element
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry for stop loss
    entry_bar = 0
    
    # Warmup: max of EMA(34) periods, RSI(14), ATR(14), volume MA(20)
    start_idx = max(34, 14, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        trend_4h_up = close_val > ema_34_4h_aligned[i]
        trend_4h_down = close_val < ema_34_4h_aligned[i]
        trend_1d_up = close_val > ema_34_1d_aligned[i]
        trend_1d_down = close_val < ema_34_1d_aligned[i]
        
        # Require both 4h and 1d trend to agree for stronger signal
        trend_up = trend_4h_up and trend_1d_up
        trend_down = trend_4h_down and trend_1d_down
        
        if position == 0:
            # Look for volume spike + extreme RSI in direction of trend
            long_signal = vol_spike and (rsi_val < 30) and trend_up
            short_signal = vol_spike and (rsi_val > 70) and trend_down
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                entry_bar = i
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                entry_bar = i
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position
            signals[i] = 0.20
            
            # Exit conditions:
            # 1. RSI returns to neutral territory (40-60)
            # 2. Stop loss: price drops 1.5*ATR below entry
            # 3. Trend flips down on either timeframe
            rsi_exit = rsi_val >= 40 and rsi_val <= 60
            stop_loss = close_val < (entry_price - 1.5 * atr[i])
            trend_exit = not (trend_4h_up and trend_1d_up)
            
            if rsi_exit or stop_loss or trend_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:  # Short position
            signals[i] = -0.20
            
            # Exit conditions:
            # 1. RSI returns to neutral territory (40-60)
            # 2. Stop loss: price rises 1.5*ATR above entry
            # 3. Trend flips up on either timeframe
            rsi_exit = rsi_val >= 40 and rsi_val <= 60
            stop_loss = close_val > (entry_price + 1.5 * atr[i])
            trend_exit = not (trend_4h_down and trend_1d_down)
            
            if rsi_exit or stop_loss or trend_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VolumeSpike_Reversal_HTFTrend_v1"
timeframe = "1h"
leverage = 1.0