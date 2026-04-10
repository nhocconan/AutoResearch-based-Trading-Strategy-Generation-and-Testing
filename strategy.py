#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND price > 50-period EMA on 1w (uptrend)
# - Short when Williams %R(14) > -20 (overbought) AND price < 50-period EMA on 1w (downtrend)
# - Volume confirmation: 1d volume > 1.3x 20-period volume SMA
# - Exit: Williams %R crosses above -50 (long) or below -50 (short)
# - Position sizing: 0.25 discrete level to balance return and drawdown
# - Target: 15-30 trades/year on 1d timeframe to stay within fee drag limits
# - Williams %R identifies extremes in bear market rallies and bull market pullbacks
# - Weekly EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals

name = "1d_williamsr_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) - momentum oscillator
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w 50-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(williams_period, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Williams %R conditions
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        # Trend filter from 1w EMA
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Exit conditions: Williams %R crosses -50 level
        exit_long = wr > -50 and wr_prev <= -50
        exit_short = wr < -50 and wr_prev >= -50
        
        if position == 0:  # Flat - look for entry
            # Long: oversold + uptrend + volume confirmation
            if wr < -80 and price_above_ema and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: overbought + downtrend + volume confirmation
            elif wr > -20 and price_below_ema and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals