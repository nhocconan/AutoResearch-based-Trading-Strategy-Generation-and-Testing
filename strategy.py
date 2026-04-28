#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when Williams %R(14) crosses above -80 (oversold) with 1d EMA34 uptrend and volume > 2x 24-bar average.
# Enter short when Williams %R(14) crosses below -20 (overbought) with 1d EMA34 downtrend and volume > 2x 24-bar average.
# Exit when Williams %R returns to -50 (mean reversion) or opposing extreme is reached.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R captures mean reversion in 6h timeframe; 1d EMA34 ensures higher timeframe alignment;
# volume spike filters weak reversals. Works in both bull (buy dips) and bear (sell rallies).

name = "6h_WilliamsR_Extremes_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2x 24-bar average volume (4 periods of 6h = 1 day)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient history for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        wr = williams_r[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (oversold), EMA34 up, volume spike
            if i >= 1 and williams_r[i-1] <= -80 and wr > -80 and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 (overbought), EMA34 down, volume spike
            elif i >= 1 and williams_r[i-1] >= -20 and wr < -20 and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            # Exit when Williams %R returns to -50 (mean reversion) or reaches -20 (overbought)
            if wr >= -50 or wr >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit
            # Exit when Williams %R returns to -50 (mean reversion) or reaches -80 (oversold)
            if wr <= -50 or wr <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when Williams %R(14) crosses above -80 (oversold) with 1d EMA34 uptrend and volume > 2x 24-bar average.
# Enter short when Williams %R(14) crosses below -20 (overbought) with 1d EMA34 downtrend and volume > 2x 24-bar average.
# Exit when Williams %R returns to -50 (mean reversion) or opposing extreme is reached.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R captures mean reversion in 6h timeframe; 1d EMA34 ensures higher timeframe alignment;
# volume spike filters weak reversals. Works in both bull (buy dips) and bear (sell rallies).

name = "6h_WilliamsR_Extremes_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2x 24-bar average volume (4 periods of 6h = 1 day)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient history for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        wr = williams_r[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (oversold), EMA34 up, volume spike
            if i >= 1 and williams_r[i-1] <= -80 and wr > -80 and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 (overbought), EMA34 down, volume spike
            elif i >= 1 and williams_r[i-1] >= -20 and wr < -20 and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            # Exit when Williams %R returns to -50 (mean reversion) or reaches -20 (overbought)
            if wr >= -50 or wr >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit
            # Exit when Williams %R returns to -50 (mean reversion) or reaches -80 (oversold)
            if wr <= -50 or wr <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals