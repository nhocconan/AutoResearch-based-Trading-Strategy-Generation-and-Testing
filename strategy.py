#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation
# - Williams %R(14) on 1d: measures overbought/oversold levels (-80 to -20 = oversold, 0 to -20 = overbought)
# - Entry logic: 
#   * Long: Williams %R < -80 (oversold) AND price > weekly EMA50 (uptrend bias) AND volume spike
#   * Short: Williams %R > -20 (overbought) AND price < weekly EMA50 (downtrend bias) AND volume spike
# - Volume confirmation: current 6h volume > 1.8x 20-period average (filters low-quality signals)
# - Weekly trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - ATR(14) trailing stop (2.5x) on 6h timeframe for risk management
# - Discrete position sizing (0.25) to minimize fee churn
# - Williams %R is effective at catching reversals in both trending and ranging markets
# - Weekly trend filter prevents counter-trend trades in strong trends
# - Volume spike confirmation ensures participation and reduces false signals
# - Target: 20-30 trades/year (80-120 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((hh_14 - ll_14) != 0, 
                         ((hh_14 - close_1d) / (hh_14 - ll_14)) * -100, 
                         -50)  # neutral when range is zero
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute 6h ATR for trailing stop
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = np.nan
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.maximum.reduce([tr1_6h, tr2_6h, tr3_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h volume and its 20-period moving average
    volume_6h = prices['volume'].values
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(volume_ma_20_6h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 6h volume for filter
        volume_6h_current = volume_6h[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Volume confirmation: current 6h volume > 1.8x 20-period average
        volume_spike = volume_6h_current > 1.8 * volume_ma_20_6h[i]
        
        # Weekly trend filter
        weekly_uptrend = close_6h[i] > ema_50_aligned[i]
        weekly_downtrend = close_6h[i] < ema_50_aligned[i]
        
        close_price = close_6h[i]
        
        if position == 0:  # Flat - look for new entries
            if volume_spike:
                # Long: Oversold AND weekly uptrend
                if oversold and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: Overbought AND weekly downtrend
                elif overbought and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_6h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_6h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
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

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation
# - Williams %R(14) on 1d: measures overbought/oversold levels (-80 to -20 = oversold, 0 to -20 = overbought)
# - Entry logic: 
#   * Long: Williams %R < -80 (oversold) AND price > weekly EMA50 (uptrend bias) AND volume spike
#   * Short: Williams %R > -20 (overbought) AND price < weekly EMA50 (downtrend bias) AND volume spike
# - Volume confirmation: current 6h volume > 1.8x 20-period average (filters low-quality signals)
# - Weekly trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - ATR(14) trailing stop (2.5x) on 6h timeframe for risk management
# - Discrete position sizing (0.25) to minimize fee churn
# - Williams %R is effective at catching reversals in both trending and ranging markets
# - Weekly trend filter prevents counter-trend trades in strong trends
# - Volume spike confirmation ensures participation and reduces false signals
# - Target: 20-30 trades/year (80-120 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((hh_14 - ll_14) != 0, 
                         ((hh_14 - close_1d) / (hh_14 - ll_14)) * -100, 
                         -50)  # neutral when range is zero
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute 6h ATR for trailing stop
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = np.nan
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.maximum.reduce([tr1_6h, tr2_6h, tr3_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h volume and its 20-period moving average
    volume_6h = prices['volume'].values
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(volume_ma_20_6h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 6h volume for filter
        volume_6h_current = volume_6h[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Volume confirmation: current 6h volume > 1.8x 20-period average
        volume_spike = volume_6h_current > 1.8 * volume_ma_20_6h[i]
        
        # Weekly trend filter
        weekly_uptrend = close_6h[i] > ema_50_aligned[i]
        weekly_downtrend = close_6h[i] < ema_50_aligned[i]
        
        close_price = close_6h[i]
        
        if position == 0:  # Flat - look for new entries
            if volume_spike:
                # Long: Oversold AND weekly uptrend
                if oversold and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: Overbought AND weekly downtrend
                elif overbought and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_6h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_6h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals