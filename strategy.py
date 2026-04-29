#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter + volume spike (>2.5x 20-period average)
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance; breakouts indicate momentum
# 1d EMA34 ensures alignment with daily trend; volume spike confirms institutional participation
# ATR-based trailing stop (2.0x) manages risk; discrete position sizing (0.25) minimizes fee churn
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in bull/bear markets: trend filter prevents counter-trend trades, volume avoids false breakouts

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike2.5x_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    # Typical price = (high + low + close) / 3
    # Range = high - low
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    typical_price = (high + low + close) / 3.0
    range_hl = high - low
    camarilla_r3 = typical_price + range_hl * 1.1 / 4.0
    camarilla_s3 = typical_price - range_hl * 1.1 / 4.0
    
    # Calculate ATR for volatility and stoploss (14-period ATR)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.5x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 20  # warmup for volume MA, Camarilla needs previous bar
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or i == 0):  # i==0 for Camarilla (needs i-1)
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Use previous bar's Camarilla levels (completed bar)
        prev_r3 = camarilla_r3[i-1]
        prev_s3 = camarilla_s3[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above R3 + price above 1d EMA34
                if curr_close > prev_r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Bearish breakout: price below S3 + price below 1d EMA34
                elif curr_close < prev_s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Track highest high since entry for trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price breaks below S3 OR ATR trailing stop hit
            if (curr_close < prev_s3 or 
                curr_close < highest_since_entry - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Track lowest low since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price breaks above R3 OR ATR trailing stop hit
            if (curr_close > prev_r3 or 
                curr_close > lowest_since_entry + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter + volume spike (>2.5x 20-period average)
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance; breakouts indicate momentum
# 1d EMA34 ensures alignment with daily trend; volume spike confirms institutional participation
# ATR-based trailing stop (2.0x) manages risk; discrete position sizing (0.25) minimizes fee churn
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in bull/bear markets: trend filter prevents counter-trend trades, volume avoids false breakouts

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike2.5x_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    # Typical price = (high + low + close) / 3
    # Range = high - low
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    typical_price = (high + low + close) / 3.0
    range_hl = high - low
    camarilla_r3 = typical_price + range_hl * 1.1 / 4.0
    camarilla_s3 = typical_price - range_hl * 1.1 / 4.0
    
    # Calculate ATR for volatility and stoploss (14-period ATR)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.5x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 20  # warmup for volume MA, Camarilla needs previous bar
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or i == 0):  # i==0 for Camarilla (needs i-1)
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Use previous bar's Camarilla levels (completed bar)
        prev_r3 = camarilla_r3[i-1]
        prev_s3 = camarilla_s3[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above R3 + price above 1d EMA34
                if curr_close > prev_r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Bearish breakout: price below S3 + price below 1d EMA34
                elif curr_close < prev_s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Track highest high since entry for trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price breaks below S3 OR ATR trailing stop hit
            if (curr_close < prev_s3 or 
                curr_close < highest_since_entry - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Track lowest low since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price breaks above R3 OR ATR trailing stop hit
            if (curr_close > prev_r3 or 
                curr_close > lowest_since_entry + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals