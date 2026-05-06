#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and ATR-scaled volume confirmation
# Long when price breaks above Camarilla R3 AND 1d close > 1d EMA50 (uptrend) AND volume > 1.5 * 20-bar ATR-scaled volume
# Short when price breaks below Camarilla S3 AND 1d close < 1d EMA50 (downtrend) AND volume > 1.5 * 20-bar ATR-scaled volume
# Exit when price retraces to Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide precise intraday support/resistance based on prior day's range
# 1d EMA50 provides strong trend filter for better regime adaptation in both bull and bear markets
# ATR-scaled volume threshold reduces false breakouts during low volatility periods

name = "4h_Camarilla_R3S3_1dEMA50_ATRVolume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 4h timeframe (based on previous 1d OHLC)
    # Need to get 1d data first for the prior day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR for volume confirmation (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled volume: volume > 1.5 * 20-bar average of (volume / ATR)
    atr_safe = np.where(atr < 1e-10, np.nan, atr)
    volume_per_atr = volume / atr_safe
    avg_volume_per_atr_20 = pd.Series(volume_per_atr).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume_per_atr > (1.5 * avg_volume_per_atr_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get prior completed 1d bar's OHLC for Camarilla calculation
        # We need the 1d bar that completed before the current 4h bar
        # Since we're on 4h timeframe, we look back to find the most recent completed 1d bar
        # We'll use the 1d data and align it properly
        
        # Get the 1d OHLC values aligned to 4h timeframe
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
        low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        
        # Use the prior completed 1d bar's OHLC (shifted by 1 to avoid look-ahead)
        # For the current 4h bar, we use the 1d bar that completed before it
        # We need to shift the aligned 1d data by the number of 4h bars in a 1d (6 bars)
        # But align_htf_to_ltf already handles the completion timing, so we shift by 1 in 1d terms
        # Actually, we want the 1d bar that is fully completed before the current 4h bar
        # Since 1d = 6 * 4h, we shift by 6 in 4h terms to get the prior 1d bar
        # But it's easier: we use the aligned 1d values and shift by 1 in 1d bar index
        # We'll calculate the Camarilla levels using the prior 1d bar's OHLC
        
        # To avoid look-ahead, we use the 1d bar that ended at least 1d before current time
        # We'll use the aligned values but reference the prior 1d bar by looking back in the 1d-aligned arrays
        # Since we don't have direct 1d bar index, we'll use a simple approach:
        # For Camarilla, we need the prior day's high, low, close
        # We'll use the 1d-aligned arrays and shift by the equivalent of 1 day in 4h bars (6 bars)
        # But to be safe and simple, we'll use the 1d bar that is confirmed closed
        
        # Actually, let's use a different approach: calculate Camarilla from the 1d data directly
        # and then align it properly with a 1-bar delay (for the 1d bar completion)
        
        # Calculate Camarilla levels from 1d data
        # Camarilla formula:
        # PP = (H + L + C) / 3
        # R3 = PP + (H - L) * 1.1 / 4
        # S3 = PP - (H - L) * 1.1 / 4
        
        # We'll calculate these for each 1d bar, then align to 4h
        # But we need to use the prior completed 1d bar, so we shift the 1d Camarilla by 1 bar
        
        # Calculate prior 1d bar's OHLC for Camarilla
        # We'll use the 1d data and shift it by 1 bar to get the prior completed bar
        if i >= 100:  # We have enough history
            # Get index of prior completed 1d bar in 1d array
            # This is complex, so let's use a rolling window on 1d data aligned to 4h
            
            # Simpler: calculate Camarilla for each 1d bar, then shift by 1 (to use prior bar)
            # Then align to 4h
            if len(df_1d) >= 2:  # Need at least 2 1d bars
                # Calculate Camarilla for each 1d bar
                H_1d = df_1d['high'].values
                L_1d = df_1d['low'].values
                C_1d = df_1d['close'].values
                
                PP_1d = (H_1d + L_1d + C_1d) / 3.0
                R3_1d = PP_1d + (H_1d - L_1d) * 1.1 / 4.0
                S3_1d = PP_1d - (H_1d - L_1d) * 1.1 / 4.0
                
                # Shift by 1 to use prior completed 1d bar (avoid look-ahead)
                PP_1d_prior = np.roll(PP_1d, 1)
                R3_1d_prior = np.roll(R3_1d, 1)
                S3_1d_prior = np.roll(S3_1d, 1)
                PP_1d_prior[0] = np.nan
                R3_1d_prior[0] = np.nan
                S3_1d_prior[0] = np.nan
                
                # Align to 4h timeframe
                PP_aligned = align_htf_to_ltf(prices, df_1d, PP_1d_prior)
                R3_aligned = align_htf_to_ltf(prices, df_1d, R3_1d_prior)
                S3_aligned = align_htf_to_ltf(prices, df_1d, S3_1d_prior)
                
                # Now we can use these aligned values
                if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
                    if position != 0:
                        signals[i] = 0.0
                        position = 0
                    continue
                
                # Trading logic
                if position == 0:
                    # Long: Break above R3 AND uptrend AND volume confirmation
                    if close[i] > R3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirmation[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short: Break below S3 AND downtrend AND volume confirmation
                    elif close[i] < S3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirmation[i]:
                        signals[i] = -0.25
                        position = -1
                elif position == 1:
                    # Exit long: Price retraces to pivot point (PP)
                    if close[i] <= PP_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                elif position == -1:
                    # Exit short: Price retraces to pivot point (PP)
                    if close[i] >= PP_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                # Not enough 1d data yet
                if position != 0:
                    signals[i] = 0.0
                    position = 0
        else:
            # Warmup period
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and ATR-scaled volume confirmation
# Long when price breaks above Camarilla R3 AND 1d close > 1d EMA50 (uptrend) AND volume > 1.5 * 20-bar ATR-scaled volume
# Short when price breaks below Camarilla S3 AND 1d close < 1d EMA50 (downtrend) AND volume > 1.5 * 20-bar ATR-scaled volume
# Exit when price retraces to Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide precise intraday support/resistance based on prior day's range
# 1d EMA50 provides strong trend filter for better regime adaptation in both bull and bear markets
# ATR-scaled volume threshold reduces false breakouts during low volatility periods

name = "4h_Camarilla_R3S3_1dEMA50_ATRVolume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR for volume confirmation (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled volume: volume > 1.5 * 20-bar average of (volume / ATR)
    atr_safe = np.where(atr < 1e-10, np.nan, atr)
    volume_per_atr = volume / atr_safe
    avg_volume_per_atr_20 = pd.Series(volume_per_atr).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume_per_atr > (1.5 * avg_volume_per_atr_20)
    
    # Calculate Camarilla levels from prior completed 1d bar
    # Camarilla formula using prior 1d's OHLC:
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 4
    # S3 = PP - (H - L) * 1.1 / 4
    
    H_1d = df_1d['high'].values
    L_1d = df_1d['low'].values
    C_1d = df_1d['close'].values
    
    PP_1d = (H_1d + L_1d + C_1d) / 3.0
    R3_1d = PP_1d + (H_1d - L_1d) * 1.1 / 4.0
    S3_1d = PP_1d - (H_1d - L_1d) * 1.1 / 4.0
    
    # Shift by 1 to use prior completed 1d bar (avoid look-ahead)
    PP_1d_prior = np.roll(PP_1d, 1)
    R3_1d_prior = np.roll(R3_1d, 1)
    S3_1d_prior = np.roll(S3_1d, 1)
    PP_1d_prior[0] = np.nan
    R3_1d_prior[0] = np.nan
    S3_1d_prior[0] = np.nan
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP_1d_prior)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_1d_prior)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_1d_prior)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 AND uptrend AND volume confirmation
            if close[i] > R3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 AND downtrend AND volume confirmation
            elif close[i] < S3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to pivot point (PP)
            if close[i] <= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to pivot point (PP)
            if close[i] >= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals