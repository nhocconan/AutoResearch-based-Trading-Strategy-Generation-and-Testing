#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour timeframe strategy using weekly pivot points combined with volume confirmation
# Weekly pivot points provide key support/resistance levels from higher timeframe
# Price rejection at weekly S1/R1 with volume confirmation indicates institutional interest
# Works in bull markets by capturing bounces from weekly support
# Works in bear markets by capturing rejections at weekly resistance
# Target: 75-150 total trades over 4 years (19-38/year)
name = "exp_14155_6h_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    pivot, r1, s1, r2, s2, r3, s3 = calculate_pivot_points(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for volume, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry signals based on weekly pivot rejection with volume
        # Long: price near weekly S1/S2 with bullish rejection + volume
        # Short: price near weekly R1/R2 with bearish rejection + volume
        near_s1 = abs(close[i] - s1_aligned[i]) <= (0.5 * atr[i])
        near_s2 = abs(close[i] - s2_aligned[i]) <= (0.5 * atr[i])
        near_r1 = abs(close[i] - r1_aligned[i]) <= (0.5 * atr[i])
        near_r2 = abs(close[i] - r2_aligned[i]) <= (0.5 * atr[i])
        
        # Bullish rejection: closing above the support level after touching it
        bullish_rejection = (near_s1 or near_s2) and (close[i] > open[i]) and vol_filter[i]
        # Bearish rejection: closing below the resistance level after touching it
        bearish_rejection = (near_r1 or near_r2) and (close[i] < open[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if bullish_rejection:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif bearish_rejection:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or bearish rejection at resistance
            if close[i] <= stop_price or bearish_rejection:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or bullish rejection at support
            if close[i] >= stop_price or bullish_rejection:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour strategy using weekly Camarilla pivot levels for mean reversion
# Weekly Camarilla levels (R3/S3, R4/S4) act as strong support/resistance
# Price reaching R3/S3 with high probability of reversal (mean reversion)
# Price breaking R4/S4 indicates strong momentum (continuation)
# Volume filter confirms institutional participation
# Works in ranging markets via reversals at R3/S3
# Works in trending markets via breakouts at R4/S4
# Target: 75-150 total trades over 4 years (19-38/year)
name = "exp_14155_6h_weekly_camarilla_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the week"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    # Camarilla levels
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Camarilla levels (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly Camarilla levels
    r3, s3, r4, s4 = calculate_camarilla(weekly_high, weekly_low, weekly_close)
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values  # Fixed: use open price for rejection signals
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for volume, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s4_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry signals based on weekly Camarilla levels
        # Mean reversion at R3/S3: price touches level and reverses
        # Continuation at R4/S4: price breaks through level with volume
        
        # Touch R3/S3 (within 0.3 * ATR)
        touch_r3 = abs(close[i] - r3_aligned[i]) <= (0.3 * atr[i])
        touch_s3 = abs(close[i] - s3_aligned[i]) <= (0.3 * atr[i])
        
        # Break R4/S4 (close beyond level)
        break_r4 = close[i] > r4_aligned[i]
        break_s4 = close[i] < s4_aligned[i]
        
        # Mean reversion signals: rejection at R3/S3
        # At R3: bearish rejection (close below open after touching resistance)
        bearish_rejection_r3 = touch_r3 and (close[i] < open_price[i]) and vol_filter[i]
        # At S3: bullish rejection (close above open after touching support)
        bullish_rejection_s3 = touch_s3 and (close[i] > open_price[i]) and vol_filter[i]
        
        # Continuation signals: breakout at R4/S4
        # Break above R4: bullish continuation
        bullish_breakout_r4 = break_r4 and vol_filter[i]
        # Break below S4: bearish continuation
        bearish_breakout_s4 = break_s4 and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if bearish_rejection_r3:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            elif bullish_rejection_s3:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif bullish_breakout_r4:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif bearish_breakout_s4:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or bearish reversal signals
            if close[i] <= stop_price or bearish_rejection_r3 or bearish_breakout_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or bullish reversal signals
            if close[i] >= stop_price or bullish_rejection_s3 or bullish_breakout_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour strategy combining weekly Supertrend for trend direction with
# 6-hour Donchian breakouts for entry timing. Weekly Supertrend filters trades
# to only take breakouts in the direction of the higher timeframe trend.
# Works in bull markets by taking long breakouts during uptrends.
# Works in bear markets by taking short breakdowns during downtrends.
# Reduces whipsaw by avoiding counter-trend trades.
# Target: 75-150 total trades over 4 years (19-38/year)
name = "exp_14155_6h_supertrend_donchian_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, atr_period, multiplier):
    """Calculate Supertrend indicator"""
    # Calculate ATR
    atr = calculate_atr(high, low, close, atr_period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    # First value
    supertrend[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        # Calculate upper and lower bands
        upper_band[i] = hl2[i] + (multiplier * atr[i])
        lower_band[i] = hl2[i] - (multiplier * atr[i])
        
        # Determine trend direction
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        # Set Supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Supertrend (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly Supertrend (10-period ATR, 3.0 multiplier)
    supertrend, trend_direction = calculate_supertrend(
        weekly_high, weekly_low, weekly_close, 10, 3.0
    )
    
    # Align weekly Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_weekly, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_weekly, trend_direction)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(supertrend_aligned[i]) or \
           np.isnan(trend_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry signals: Donchian breakout in direction of weekly Supertrend trend
        # Long: break above upper band + weekly uptrend + volume
        # Short: break below lower band + weekly downtrend + volume
        breakout_long = (close[i] > highest_high[i-1]) and (trend_aligned[i] == 1) and vol_filter[i]
        breakout_short = (close[i] < lowest_low[i-1]) and (trend_aligned[i] == -1) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or breakdown of lower band
            if close[i] <= stop_price or close[i] < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or breakout of upper band
            if close[i] >= stop_price or close[i] > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour strategy using weekly Elder Ray (Bull Power/Bear Power) for
# assessing market strength combined with 6-hour price action for entry.
# Bull Power = High - EMA13 (measures bull strength)
# Bear Power = EMA13 - Low (measures bear strength)
# When Bull Power > 0 and increasing, bulls are in control
# When Bear Power > 0 and increasing, bears are in control
# Enter long on bullish exhaustion (Bear Power peaking) with volume
# Enter short on bearish exhaustion (Bull Power peaking) with volume
# Works in trending markets by catching pullbacks in direction of trend
# Works in ranging markets by fading extremes
# Target: 75-150 total trades over 4 years (19-38/year)
name = "exp_14155_6h_weekly_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_elder_ray(high, low, close, ema_period):
    """Calculate Elder Ray indicators: Bull Power and Bear Power"""
    ema = calculate_ema(close, ema_period)
    bull_power = high - ema
    bear_power = ema - low
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Elder Ray (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly Elder Ray (13-period EMA)
    bull_power, bear_power = calculate_elder_ray(
        weekly_high, weekly_low, weekly_close, 13
    )
    
    # Align weekly Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_weekly, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_weekly, bear_power)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for volume, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry signals based on Elder Ray exhaustion
        # Look for peaking Bull Power (bearish exhaustion) or peaking Bear Power (bullish exhaustion)
        
        # Calculate rate of change to detect peaks
        if i >= 2:
            bull_power_roc = (bull_power_aligned[i] - bull_power_aligned[i-2]) / 2
            bear_power_roc = (bear_power_aligned[i] - bear_power_aligned[i-2]) / 2
        else:
            bull_power_roc = 0
            bear_power_roc = 0
        
        # Bullish exhaustion: Bear Power peaking (starting to decline after rise)
        bearish_exhaustion = (
            bear_power_aligned[i] > 0 and  # Bears have been in control
            bear_power_roc < 0 and         # Bear Power is declining (peaking)
            close[i] < open_price[i] and   # Bearish candle
            vol_filter[i]
        )
        
        # Bearish exhaustion: Bull Power peaking (starting to decline after rise)
        bullish_exhaustion = (
            bull_power_aligned[i] > 0 and  # Bulls have been in control
            bull_power_roc < 0 and         # Bull Power is declining (peaking)
            close[i] > open_price[i] and   # Bullish candle
            vol_filter[i]
        )
        
        # Generate signals
        if position == 0:
            if bearish_exhaustion:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            elif bullish_exhaustion:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or bullish exhaustion signal
            if close[i] <= stop_price or bullish_exhaustion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or bearish exhaustion signal
            if close[i] >= stop_price or bearish_exhaustion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

}