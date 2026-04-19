#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ADX_StochRSI_Momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate StochRSI(14) on 1d close
    close_1d = df_1d['close'].values
    rsi_period = 14
    stoch_period = 14
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).min().values
    rsi_max = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).max().values
    stoch_rsi = np.where((rsi_max - rsi_min) != 0, (rsi - rsi_min) / (rsi_max - rsi_min) * 100, 50)
    
    # Align indicators to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    stoch_rsi_aligned = align_htf_to_ltf(prices, df_1d, stoch_rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(stoch_rsi_aligned[i])):
            signals[i] = 0.0
            continue
            
        # ADX filter: only trade when trend is strong (ADX > 25)
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long when StochRSI < 20 (oversold) and trend is up (+DI > -DI)
            if (stoch_rsi_aligned[i] < 20 and 
                trend_filter and
                plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short when StochRSI > 80 (overbought) and trend is down (-DI > +DI)
            elif (stoch_rsi_aligned[i] > 80 and 
                  trend_filter and
                  minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when StochRSI > 80 (overbought) or trend weakens
            if (stoch_rsi_aligned[i] > 80 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when StochRSI < 20 (oversold) or trend weakens
            if (stoch_rsi_aligned[i] < 20 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: plus_di_aligned and minus_di_aligned need to be defined
# Let me fix this by calculating and aligning them properly

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ADX_StochRSI_Momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate StochRSI(14) on 1d close
    close_1d = df_1d['close'].values
    rsi_period = 14
    stoch_period = 14
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).min().values
    rsi_max = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).max().values
    stoch_rsi = np.where((rsi_max - rsi_min) != 0, (rsi - rsi_min) / (rsi_max - rsi_min) * 100, 50)
    
    # Align indicators to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di, additional_delay_bars=0)
    minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di, additional_delay_bars=0)
    stoch_rsi_aligned = align_htf_to_ltf(prices, df_1d, stoch_rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(stoch_rsi_aligned[i])):
            signals[i] = 0.0
            continue
            
        # ADX filter: only trade when trend is strong (ADX > 25)
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long when StochRSI < 20 (oversold) and trend is up (+DI > -DI)
            if (stoch_rsi_aligned[i] < 20 and 
                trend_filter and
                plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short when StochRSI > 80 (overbought) and trend is down (-DI > +DI)
            elif (stoch_rsi_aligned[i] > 80 and 
                  trend_filter and
                  minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when StochRSI > 80 (overbought) or trend weakens
            if (stoch_rsi_aligned[i] > 80 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when StochRSI < 20 (oversold) or trend weakens
            if (stoch_rsi_aligned[i] < 20 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals