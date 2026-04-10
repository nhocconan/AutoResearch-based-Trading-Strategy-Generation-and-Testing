#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1d close > 1d EMA50 (uptrend) AND 6h volume > 1.5x 20-period average
# - Short when Williams %R(14) > -20 (overbought) AND 1d close < 1d EMA50 (downtrend) AND 6h volume > 1.5x 20-period average
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R identifies exhaustion points in both trending and ranging markets
# - Daily EMA50 filter ensures we trade with higher timeframe trend (avoids counter-trend in strong moves)
# - Volume confirmation reduces false signals from low-participation exhaustion
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h Williams %R(14)
    highest_high = np.full_like(high, np.nan, dtype=float)
    lowest_low = np.full_like(low, np.nan, dtype=float)
    for i in range(13, len(high)):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Avoid division by zero
    
    # Pre-compute 6h volume MA(20)
    vol_ma_6h = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_ma_6h[i] = np.mean(volume[i-19:i+1])
    
    # Pre-compute 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * multiplier) + (ema_50_1d[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R is 6h indicator, no HTF alignment needed
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_6h)    # Actually 6h, but using mtf_data for consistency
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Correct alignment: 6h indicators don't need HTF alignment, 1d EMA does
    williams_r_aligned = williams_r  # 6h indicator, already aligned
    vol_ma_6h_aligned = vol_ma_6h    # 6h indicator, already aligned
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)  # 1d EMA needs alignment
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup for Williams %R(14) and EMA50
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.5x average)
            vol_spike = volume[i] > 1.5 * vol_ma_6h_aligned[i]
            
            # Long conditions: Williams %R < -80 (oversold) AND 1d uptrend (close > EMA50) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R > -20 (overbought) AND 1d downtrend (close < EMA50) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            exit_long = (position == 1 and williams_r_aligned[i] > -50)
            exit_short = (position == -1 and williams_r_aligned[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1d close > 1d EMA50 (uptrend) AND 6h volume > 1.5x 20-period average
# - Short when Williams %R(14) > -20 (overbought) AND 1d close < 1d EMA50 (downtrend) AND 6h volume > 1.5x 20-period average
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R identifies exhaustion points in both trending and ranging markets
# - Daily EMA50 filter ensures we trade with higher timeframe trend (avoids counter-trend in strong moves)
# - Volume confirmation reduces false signals from low-participation exhaustion
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTf data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h Williams %R(14)
    highest_high = np.full_like(high, np.nan, dtype=float)
    lowest_low = np.full_like(low, np.nan, dtype=float)
    for i in range(13, len(high)):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Avoid division by zero
    
    # Pre-compute 6h volume MA(20)
    vol_ma_6h = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_ma_6h[i] = np.mean(volume[i-19:i+1])
    
    # Pre-compute 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * multiplier) + (ema_50_1d[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = williams_r  # 6h indicator, already aligned
    vol_ma_6h_aligned = vol_ma_6h    # 6h indicator, already aligned
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)  # 1d EMA needs alignment
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup for Williams %R(14) and EMA50
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.5x average)
            vol_spike = volume[i] > 1.5 * vol_ma_6h_aligned[i]
            
            # Long conditions: Williams %R < -80 (oversold) AND 1d uptrend (close > EMA50) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R > -20 (overbought) AND 1d downtrend (close < EMA50) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            exit_long = (position == 1 and williams_r_aligned[i] > -50)
            exit_short = (position == -1 and williams_r_aligned[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals