#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d volume spike
# - Primary: 1h RSI(14) < 30 for long, > 70 for short (mean reversion)
# - HTF trend: 4h close > 20-period EMA for long bias, < 20-period EMA for short bias
# - HTF volume: 1d volume > 1.5x 20-period MA for participation confirmation
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Long: RSI<30 + 4h EMA uptrend + 1d volume spike + session
# - Short: RSI>70 + 4h EMA downtrend + 1d volume spike + session
# - Exit: RSI crosses 50 (mean reversion midpoint)
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# - Works in bull/bear: RSI mean reversion works in ranges, 4h EMA filter avoids counter-trend trades, volume confirms participation

name = "1h_4h_1d_rsi_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_4h = df_4h['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA(20)
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(30, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: RSI<30 + 4h EMA uptrend + volume spike + session
            if (rsi[i] < 30 and close[i] > ema_4h_aligned[i] and volume_confirm):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI>70 + 4h EMA downtrend + volume spike + session
            elif (rsi[i] > 70 and close[i] < ema_4h_aligned[i] and volume_confirm):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: RSI crosses 50 (mean reversion midpoint)
            if position == 1:  # Long position
                if rsi[i] >= 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if rsi[i] <= 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals