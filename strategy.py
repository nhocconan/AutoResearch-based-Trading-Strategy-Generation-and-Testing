#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Camarilla H3/L3 OR price crosses 12h EMA50 OR ATR stoploss (2.0 * ATR)
# Uses Camarilla pivot levels from 1d HTF for institutional support/resistance
# Volume spike confirms breakout validity, EMA50 filter reduces whipsaws
# Discrete position sizing: 0.30 for long/short, 0.0 for flat to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels from previous 1d bar
        if i >= 20:  # Need at least one completed 1d bar (approximate)
            # Use previous 1d OHLC for Camarilla calculation
            # Since we don't have direct 1d OHLC in loop, approximate using 4h data over 1d period
            # Simplified: use rolling 96-period (4h * 6 = 24h) for daily OHLC approximation
            lookback = min(96, i)  # Use available data
            if lookback >= 96:
                # Approximate daily OHLC from last 96 4h bars (24 hours)
                day_high = np.max(high[i-96:i])
                day_low = np.min(low[i-96:i])
                day_close = close[i-1]  # Previous bar close as daily close approximation
                day_open = open_prices[i-96] if 'open' in prices.columns and i-96 >= 0 else close[i-96]
            else:
                # Fallback to simpler calculation
                day_high = np.max(high[max(0, i-24):i])  # Last 24h approximation
                day_low = np.min(low[max(0, i-24):i])
                day_close = close[i-1]
                day_open = open_prices[max(0, i-24)] if 'open' in prices.columns and i-24 >= 0 else close[max(0, i-24)]
            
            # Calculate Camarilla levels
            range_val = day_high - day_low
            camarilla_h5 = day_close + 1.1 * range_val * 1.1 / 2
            camarilla_h4 = day_close + 1.1 * range_val * 1.1 / 4
            camarilla_h3 = day_close + 1.1 * range_val * 1.1 / 6
            camarilla_l3 = day_close - 1.1 * range_val * 1.1 / 6
            camarilla_l4 = day_close - 1.1 * range_val * 1.1 / 4
            camarilla_l5 = day_close - 1.1 * range_val * 1.1 / 2
        else:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Close below Camarilla H3 OR price below 12h EMA50 OR stoploss hit
            if curr_close < camarilla_h3 or curr_close < curr_ema_12h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Camarilla L3 OR price above 12h EMA50 OR stoploss hit
            if curr_close > camarilla_l3 or curr_close > curr_ema_12h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: Close > Camarilla H3 AND price > 12h EMA50 AND volume spike
            if (curr_close > camarilla_h3 and 
                curr_close > curr_ema_12h and
                vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: Close < Camarilla L3 AND price < 12h EMA50 AND volume spike
            elif (curr_close < camarilla_l3 and 
                  curr_close < curr_ema_12h and
                  vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Camarilla H3/L3 OR price crosses 12h EMA50 OR ATR stoploss (2.0 * ATR)
# Uses Camarilla pivot levels from 1d HTF for institutional support/resistance
# Volume spike confirms breakout validity, EMA50 filter reduces whipsaws
# Discrete position sizing: 0.30 for long/short, 0.0 for flat to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values if 'open' in prices.columns else close  # fallback
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels from previous completed 1d bar
        # Need to find the most recent completed 1d bar
        # Approximate: 1d = 96 * 4h bars
        bars_1d = 96
        if i >= bars_1d:
            # Use OHLC from the completed 1d bar ending at i - (i % bars_1d)
            # Simplified: use the 1d bar that ended bars_1d periods ago
            start_idx_1d = i - bars_1d
            end_idx_1d = i
            if start_idx_1d >= 0:
                day_high = np.max(high[start_idx_1d:end_idx_1d])
                day_low = np.min(low[start_idx_1d:end_idx_1d])
                day_close = close[end_idx_1d - 1]  # Close of the completed 1d bar
                day_open = open_prices[start_idx_1d]
            else:
                # Not enough data for full 1d bar, use available
                day_high = np.max(high[:i])
                day_low = np.min(low[:i])
                day_close = close[i-1] if i > 0 else close[0]
                day_open = open_prices[0]
        else:
            # Not enough data for 1d bar, use all available
            day_high = np.max(high[:i]) if i > 0 else high[0]
            day_low = np.min(low[:i]) if i > 0 else low[0]
            day_close = close[i-1] if i > 0 else close[0]
            day_open = open_prices[0] if i > 0 else open_prices[0]
        
        # Calculate Camarilla levels
        range_val = day_high - day_low
        camarilla_h3 = day_close + 1.1 * range_val * 1.1 / 6
        camarilla_l3 = day_close - 1.1 * range_val * 1.1 / 6
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Close below Camarilla H3 OR price below 12h EMA50 OR stoploss hit
            if curr_close < camarilla_h3 or curr_close < curr_ema_12h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Camarilla L3 OR price above 12h EMA50 OR stoploss hit
            if curr_close > camarilla_l3 or curr_close > curr_ema_12h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: Close > Camarilla H3 AND price > 12h EMA50 AND volume spike
            if (curr_close > camarilla_h3 and 
                curr_close > curr_ema_12h and
                vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: Close < Camarilla L3 AND price < 12h EMA50 AND volume spike
            elif (curr_close < camarilla_l3 and 
                  curr_close < curr_ema_12h and
                  vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Camarilla H3/L3 OR price crosses 12h EMA50 OR ATR stoploss (2.0 * ATR)
# Uses Camarilla pivot levels from 1d HTF for institutional support/resistance
# Volume spike confirms breakout validity, EMA50 filter reduces whipsaws
# Discrete position sizing: 0.30 for long/short, 0.0 for flat to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values if 'open' in prices.columns else close  # fallback
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels from previous completed 1d bar
        # Need to find the most recent completed 1d bar
        # Approximate: 1d = 96 * 4h bars
        bars_1d = 96
        if i >= bars_1d:
            # Use OHLC from the completed 1d bar ending at i - (i % bars_1d)
            # Simplified: use the 1d bar that ended bars_1d periods ago
            start_idx_1d = i - bars_1d
            end_idx_1d = i
            if start_idx_1d >= 0:
                day_high = np.max(high[start_idx_1d:end_idx_1d])
                day_low = np.min(low[start_idx_1d:end_idx_1d])
                day_close = close[end_idx_1d - 1]  # Close of the completed 1d bar
                day_open = open_prices[start_idx_1d]
            else:
                # Not enough data for full 1d bar, use available
                day_high = np.max(high[:i])
                day_low = np.min(low[:i])
                day_close = close[i-1] if i > 0 else close[0]
                day_open = open_prices[0]
        else:
            # Not enough data for 1d bar, use all available
            day_high = np.max(high[:i]) if i > 0 else high[0]
            day_low = np.min(low[:i]) if i > 0 else low[0]
            day_close = close[i-1] if i > 0 else close[0]
            day_open = open_prices[0] if i > 0 else open_prices[0]
        
        # Calculate Camarilla levels
        range_val = day_high - day_low
        camarilla_h3 = day_close + 1.1 * range_val * 1.1 / 6
        camarilla_l3 = day_close - 1.1 * range_val * 1.1 / 6
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Close below Camarilla H3 OR price below 12h EMA50 OR stoploss hit
            if curr_close < camarilla_h3 or curr_close < curr_ema_12h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Camarilla L3 OR price above 12h EMA50 OR stoploss hit
            if curr_close > camarilla_l3 or curr_close > curr_ema_12h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: Close > Camarilla H3 AND price > 12h EMA50 AND volume spike
            if (curr_close > camarilla_h3 and 
                curr_close > curr_ema_12h and
                vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: Close < Camarilla L3 AND price < 12h EMA50 AND volume spike
            elif (curr_close < camarilla_l3 and 
                  curr_close < curr_ema_12h and
                  vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals