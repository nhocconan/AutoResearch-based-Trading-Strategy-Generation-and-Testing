#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and 1d volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (market structure bullish) AND 12h EMA(50) rising AND 1d volume > 1.5x 20-period average
# - Short when Bear Power > 0 AND Bull Power < 0 (market structure bearish) AND 12h EMA(50) falling AND 1d volume > 1.5x 20-period average
# - Exit when Elder Power signals reverse (Bull Power < 0 for long, Bear Power < 0 for short) OR volume drops below average
# - Uses discrete position sizing 0.25 to limit fee churn
# - Elder Ray measures price relative to EMA, showing true bull/bear power beyond just price action
# - 12h EMA(50) trend filter ensures we trade with intermediate-term momentum
# - Volume confirmation reduces false signals from weak participation
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_1d_elder_ray_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 60 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema_13  # High minus EMA(13)
    bear_power = ema_13 - low   # EMA(13) minus Low
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_12h_rising = np.full_like(ema_50_12h, False, dtype=bool)
    ema_50_12h_falling = np.full_like(ema_50_12h, False, dtype=bool)
    for i in range(1, len(ema_50_12h)):
        if not np.isnan(ema_50_12h[i]) and not np.isnan(ema_50_12h[i-1]):
            ema_50_12h_rising[i] = ema_50_12h[i] > ema_50_12h[i-1]
            ema_50_12h_falling[i] = ema_50_12h[i] < ema_50_12h[i-1]
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align HTF indicators to 6h timeframe
    ema_50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_rising)
    ema_50_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_falling)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.5x average)
            vol_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
            
            # Long conditions: Bull Power > 0 AND Bear Power < 0 (bullish structure) 
            #              AND 12h EMA(50) rising AND volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                ema_50_12h_rising_aligned[i] and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND Bull Power < 0 (bearish structure)
            #               AND 12h EMA(50) falling AND volume spike
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  ema_50_12h_falling_aligned[i] and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Power signals reverse OR volume drops below average
            vol_normal = volume[i] <= vol_ma_1d_aligned[i]
            exit_long = (position == 1 and (bull_power[i] < 0 or vol_normal))
            exit_short = (position == -1 and (bear_power[i] < 0 or vol_normal))
            
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

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and 1d volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (market structure bullish) AND 12h EMA(50) rising AND 1d volume > 1.5x 20-period average
# - Short when Bear Power > 0 AND Bull Power < 0 (market structure bearish) AND 12h EMA(50) falling AND 1d volume > 1.5x 20-period average
# - Exit when Elder Power signals reverse (Bull Power < 0 for long, Bear Power < 0 for short) OR volume drops below average
# - Uses discrete position sizing 0.25 to limit fee churn
# - Elder Ray measures price relative to EMA, showing true bull/bear power beyond just price action
# - 12h EMA(50) trend filter ensures we trade with intermediate-term momentum
# - Volume confirmation reduces false signals from weak participation
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_1d_elder_ray_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 60 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema_13  # High minus EMA(13)
    bear_power = ema_13 - low   # EMA(13) minus Low
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_12h_rising = np.full_like(ema_50_12h, False, dtype=bool)
    ema_50_12h_falling = np.full_like(ema_50_12h, False, dtype=bool)
    for i in range(1, len(ema_50_12h)):
        if not np.isnan(ema_50_12h[i]) and not np.isnan(ema_50_12h[i-1]):
            ema_50_12h_rising[i] = ema_50_12h[i] > ema_50_12h[i-1]
            ema_50_12h_falling[i] = ema_50_12h[i] < ema_50_12h[i-1]
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align HTF indicators to 6h timeframe
    ema_50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_rising)
    ema_50_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_falling)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.5x average)
            vol_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
            
            # Long conditions: Bull Power > 0 AND Bear Power < 0 (bullish structure) 
            #              AND 12h EMA(50) rising AND volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                ema_50_12h_rising_aligned[i] and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND Bull Power < 0 (bearish structure)
            #               AND 12h EMA(50) falling AND volume spike
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  ema_50_12h_falling_aligned[i] and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Power signals reverse OR volume drops below average
            vol_normal = volume[i] <= vol_ma_1d_aligned[i]
            exit_long = (position == 1 and (bull_power[i] < 0 or vol_normal))
            exit_short = (position == -1 and (bear_power[i] < 0 or vol_normal))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals