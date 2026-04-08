# 4h_vwap_reversion_1d_trend_volume_v1
# Hypothesis: Mean reversion to VWAP on 4h timeframe, filtered by 1d trend (EMA50) and volume confirmation.
# In bull markets, buy pullbacks to VWAP in uptrend; in bear markets, sell rallies to VWAP in downtrend.
# Volume filter ensures institutional participation, reducing false signals.
# Target: 20-40 trades/year via VWAP reversion + 1d trend filter + volume confirmation.

name = "4h_vwap_reversion_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # VWAP calculation (cumulative)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.zeros_like(vwap_num), where=vwap_den!=0)
    
    # Price deviation from VWAP (as percentage)
    vwap_deviation = (close - vwap) / vwap * 100
    
    # RSI for momentum confirmation (14-period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average volume
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
    ema_daily[ema_period-1] = np.mean(close_daily[:ema_period])
    for i in range(ema_period, len(close_daily)):
        ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to 4h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(20, rsi_period, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(vwap_deviation[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema_daily_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price returns to VWAP or momentum fails
            if vwap_deviation[i] >= 0 or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price returns to VWAP or momentum fails
            if vwap_deviation[i] <= 0 or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price below VWAP, RSI momentum, volume, and HTF uptrend
            if (vwap_deviation[i] < -1.0 and  # Price below VWAP by at least 1%
                rsi[i] > 30 and rsi[i] < 50 and  # Not oversold, but bullish momentum
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: price above VWAP, RSI momentum, volume, and HTF downtrend
            elif (vwap_deviation[i] > 1.0 and   # Price above VWAP by at least 1%
                  rsi[i] > 50 and rsi[i] < 70 and  # Not overbought, but bearish momentum
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals