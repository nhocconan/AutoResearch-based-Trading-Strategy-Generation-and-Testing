#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Uses 12h EMA > EMA50 to identify bullish bias and EMA < EMA50 for bearish bias.
# Long when price breaks above R3 AND 12h EMA > EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND 12h EMA < EMA50 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years.
# Volume spike threshold set to 2.0x to reduce false breakouts and improve signal quality.
# Designed to work in bull markets (trend continuation) and bear markets (mean reversion at extremes).

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Camarilla levels (based on previous 12h bar's range)
    # We need the previous completed 12h bar for each 6h bar
    daily_high = df_12h['high'].values
    daily_low = df_12h['low'].values
    daily_close = df_12h['close'].values
    
    camarilla_r3_12h = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3_12h = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (use previous 12h bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below S3
        
        # Trend filter: EMA > EMA50 for bullish bias, EMA < EMA50 for bearish bias
        bullish_bias = ema_aligned[i] > close_12h[-1] if i == len(prices)-1 else ema_aligned[i] > close[i]  # Simplified: use current price vs EMA
        bearish_bias = ema_aligned[i] < close_12h[-1] if i == len(prices)-1 else ema_aligned[i] < close[i]
        
        # Actually, we need to compare 6h price to 12h EMA - but we don't have 6h EMA
        # Instead, use the 12h EMA aligned to 6h as the trend filter directly
        # For simplicity, we'll use: if current 6h close > aligned 12h EMA50 = bullish
        # But we don't have 6h close in 12h EMA terms. Let's use a different approach.
        
        # Correct approach: use the 12h EMA50 value directly as trend filter
        # Since we aligned it, we can use it as is - but we need to compare to something
        # Let's use the 12h close price as reference, but aligned
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        bullish_bias = close_12h_aligned[i] > ema_aligned[i]  # 12h price above its EMA50 = bullish
        bearish_bias = close_12h_aligned[i] < ema_aligned[i]  # 12h price below its EMA50 = bearish
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND bullish bias AND volume confirmation
            if (breakout_up and 
                bullish_bias and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND bearish bias AND volume confirmation
            elif (breakout_down and 
                  bearish_bias and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR bearish bias (trend change)
            if (curr_low < camarilla_s3_aligned[i] or 
                bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR bullish bias (trend change)
            if (curr_high > camarilla_r3_aligned[i] or 
                bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Uses 12h EMA > EMA50 to identify bullish bias and EMA < EMA50 for bearish bias.
# Long when price breaks above R3 AND 12h EMA > EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND 12h EMA < EMA50 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years.
# Volume spike threshold set to 2.0x to reduce false breakouts and improve signal quality.
# Designed to work in bull markets (trend continuation) and bear markets (mean reversion at extremes).

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Camarilla levels (based on previous 12h bar's range)
    # We need the previous completed 12h bar for each 6h bar
    daily_high = df_12h['high'].values
    daily_low = df_12h['low'].values
    daily_close = df_12h['close'].values
    
    camarilla_r3_12h = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3_12h = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (use previous 12h bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below S3
        
        # Trend filter: use 12h price vs its EMA50 for bias
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        bullish_bias = close_12h_aligned[i] > ema_aligned[i]  # 12h price above its EMA50 = bullish
        bearish_bias = close_12h_aligned[i] < ema_aligned[i]  # 12h price below its EMA50 = bearish
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND bullish bias AND volume confirmation
            if (breakout_up and 
                bullish_bias and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND bearish bias AND volume confirmation
            elif (breakout_down and 
                  bearish_bias and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR bearish bias (trend change)
            if (curr_low < camarilla_s3_aligned[i] or 
                bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR bullish bias (trend change)
            if (curr_high > camarilla_r3_aligned[i] or 
                bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals