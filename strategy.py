#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# - Primary: 1h RSI(14) < 30 for long, > 70 for short (mean reversion in ranging markets)
# - HTF trend: 4h EMA(50) slope determines market regime (rising = long bias, falling = short bias)
# - HTF volume: 4h volume > 1.3x 20-period MA for institutional participation
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Long: RSI < 30 + rising 4h EMA slope + volume spike + session
# - Short: RSI > 70 + falling 4h EMA slope + volume spike + session
# - Exit: RSI crosses 50 (mean reversion complete) or EMA slope reverses
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Target: 80-180 total trades over 4 years (20-45/year) for 1h timeframe
# - Works in bull/bear: RSI captures mean reversion in ranges, EMA slope filters trend strength, volume confirms participation

name = "1h_4h_rsi_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA(50) and its slope
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate EMA slope (rate of change over 3 periods)
    ema_slope = np.zeros_like(ema_4h_aligned)
    for i in range(3, len(ema_4h_aligned)):
        if not np.isnan(ema_4h_aligned[i]) and not np.isnan(ema_4h_aligned[i-3]):
            ema_slope[i] = (ema_4h_aligned[i] - ema_4h_aligned[i-3]) / 3
        else:
            ema_slope[i] = np.nan
    
    # Calculate 4h volume MA(20)
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(volume_ma_20_4h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period MA
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        volume_confirm = volume_4h_aligned[i] > 1.3 * volume_ma_20_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: RSI < 30 + rising 4h EMA slope + volume spike + session
            if (rsi[i] < 30 and ema_slope[i] > 0 and volume_confirm):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 + falling 4h EMA slope + volume spike + session
            elif (rsi[i] > 70 and ema_slope[i] < 0 and volume_confirm):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: RSI crosses 50 (mean reversion complete) or EMA slope reverses
            if position == 1:  # Long position
                if rsi[i] > 50 or ema_slope[i] < 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if rsi[i] < 50 or ema_slope[i] > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals