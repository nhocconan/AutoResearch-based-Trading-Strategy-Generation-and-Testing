#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA(50) trend filter and volume confirmation.
# Uses daily EMA for trend direction to avoid counter-trend trades, works in bull markets by capturing momentum
# breakouts and in bear markets by using daily EMA to avoid false breakdowns. Volume filter ensures institutional
# participation. Target: 75-200 trades over 4 years.
name = "exp_14177_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA(50) trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
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
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA, 20 for volume, 14 for ATR)
    start = max(20, 50, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_aligned[i]) or \
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
        
        # Donchian breakout signals with volume and EMA filter
        # Long: break above upper band + above 1d EMA + volume
        # Short: break below lower band + below 1d EMA + volume
        breakout_long = (close[i] > highest_high[i-1]) and (close[i] > ema_50_aligned[i]) and vol_filter[i]
        breakout_short = (close[i] < lowest_low[i-1]) and (close[i] < ema_50_aligned[i]) and vol_filter[i]
        
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

# Hypothesis: 4-hour Camarilla pivot levels from 1d + volume spike + choppiness regime filter.
# Uses 1-day OHLC to calculate Camarilla levels (support/resistance) for mean reversion in ranging markets
# and breakout in trending markets. Volume spike confirms institutional interest. Choppiness filter
# avoids whipsaws in low-volatility environments. Target: 75-200 trades over 4 years.
name = "exp_14177_4h_camarilla_vol_chop_v1"
timeframe = "4h"
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6, L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12, L1 = close - 1.1*(high-low)/12
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_h2 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l2 = close_1d - 1.1 * (high_1d - low_1d) / 6
    camarilla_h1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_l1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period) - range detection
    # CHOP = 100 * log10(sum(TR) / (ATR * n)) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (atr_14 * 14)) / np.log10(14)
    
    # Volume filter: volume > 2.0x 20-period average (strong institutional interest)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 14 for chop, 20 for volume, 14 for ATR)
    start = max(14, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(chop[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Trading logic:
        # In ranging markets (CHOP > 61.8): mean reversion at H3/L3
        # In trending markets (CHOP < 38.2): breakout at H4/L4
        # Chop between 38.2-61.8: no trade (transition zone)
        
        if chop[i] > 61.8:  # Ranging market - mean reversion
            # Long near L3 with volume
            if close[i] <= l3_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short near H3 with volume
            elif close[i] >= h3_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            # Exit on opposite touch
            elif position == 1 and close[i] >= h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] <= l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position * 0.25 if position != 0 else 0.0
                
        elif chop[i] < 38.2:  # Trending market - breakout
            # Long breakout above H4 with volume
            if close[i] > h4_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short breakdown below L4 with volume
            elif close[i] < l4_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            # Exit on retracement to H2/L2
            elif position == 1 and close[i] < h2_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > l2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position * 0.25 if position != 0 else 0.0
                
        else:  # Transition zone - no trade
            signals[i] = 0.0
            if position != 0:
                # Exit on stop loss only in transition zone
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour TRIX(9) + volume spike + choppiness regime filter.
# TRIX filters out insignificant price movements and shows smoothed momentum.
# Works in trending markets (CHOP < 38.2) by following TRIX crosses.
# Works in ranging markets (CHOP > 61.8) by fading extremes at TRIX overbought/oversold.
# Volume spike confirms institutional participation. Target: 75-200 trades over 4 years.
name = "exp_14177_4h_trix_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def ema(series, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX(9): triple EMA of percent change
    roc = np.diff(close, prepend=close[0]) / close  # Rate of change
    ema1 = ema(roc, 9)
    ema2 = ema(ema1, 9)
    ema3 = ema(ema2, 9)
    trix = ema3 * 100  # Convert to percentage
    
    # Choppiness Index (14-period) - range detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (atr_14 * 14)) / np.log10(14)
    
    # Volume filter: volume > 2.0x 20-period average (strong institutional interest)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 9*3 for TRIX, 14 for chop, 20 for volume, 14 for ATR)
    start = max(27, 14, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Trading logic based on regime:
        # Trending market (CHOP < 38.2): follow TRIX crosses
        # Ranging market (CHOP > 61.8): fade TRIX extremes
        # Transition zone: no trade
        
        if chop[i] < 38.2:  # Trending market
            # Long when TRIX crosses above zero with volume
            if trix[i] > 0 and trix[i-1] <= 0 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short when TRIX crosses below zero with volume
            elif trix[i] < 0 and trix[i-1] >= 0 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            # Exit on opposite cross
            elif position == 1 and trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position * 0.25 if position != 0 else 0.0
                
        elif chop[i] > 61.8:  # Ranging market
            # Long when TRIX oversold (< -0.5) and turning up with volume
            if trix[i] < -0.5 and trix[i] > trix[i-1] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short when TRIX overbought (> 0.5) and turning down with volume
            elif trix[i] > 0.5 and trix[i] < trix[i-1] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            # Exit when TRIX returns to neutral zone
            elif position == 1 and trix[i] > 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and trix[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position * 0.25 if position != 0 else 0.0
                
        else:  # Transition zone - no trade
            signals[i] = 0.0
            if position != 0:
                # Exit on stop loss only in transition zone
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Connors RSI (CRSI) + volume spike + choppiness regime filter.
# CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 captures short-term momentum exhaustion.
# Works in ranging markets (CHOP > 61.8) by fading extremes (CRSI < 10 long, > 90 short).
# Works in trending markets (CHOP < 38.2) by pulling back to rising/falling EMA(20).
# Volume spike confirms institutional participation. Target: 75-200 trades over 4 years.
name = "exp_14177_4h_crsi_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def rsi(series, period):
    """Calculate RSI with proper min_periods"""
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Connors RSI components
    rsi_3 = rsi(close, 3)
    
    # Streak RSI: RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    # RSI of streak values (capped at +/- 25 for extreme streaks)
    streak_clipped = np.clip(streak, -25, 25)
    rsi_streak = rsi(streak_clipped, 2)
    
    # Percent Rank of close over 100 periods
    def percent_rank(series, window):
        rank = np.zeros_like(series)
        for i in range(len(series)):
            if i < window:
                rank[i] = np.nan
            else:
                window_data = series[i-window:i]
                rank[i] = np.sum(window_data <= series[i]) / window * 100
        return rank
    percent_rank_100 = percent_rank(close, 100)
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = (rsi_3 + rsi_streak + percent_rank_100) / 3
    
    # Choppiness Index (14-period) - range detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (atr_14 * 14)) / np.log10(14)
    
    # Volume filter: volume > 2.0x 20-period average (strong institutional interest)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    # EMA(20) for trend filter in trending markets
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 3 for RSI, 2 for streak, 100 for percent rank, 14 for chop, 20 for volume/ema, 14 for ATR)
    start = max(3, 2, 100, 14, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_20[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Trading logic based on regime:
        # Ranging market (CHOP > 61.8): mean reversion at CRSI extremes
        # Trending market (CHOP < 38.2): pullback to EMA(20)
        # Transition zone: no trade
        
        if chop[i] > 61.8:  # Ranging market
            # Long when CRSI oversold (< 10) with volume
            if crsi[i] < 10 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short when CRSI overbought (> 90) with volume
            elif crsi[i] > 90 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            # Exit when CRSI returns to neutral (40-60)
            elif position == 1 and crsi[i] > 60:
                signals[i] = 0.0
                position = 0
            elif position == -1 and crsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position * 0.25 if position != 0 else 0.0
                
        elif chop[i] < 38.2:  # Trending market
            # Long on pullback to EMA(20) in uptrend with volume
            if close[i] <= ema_20[i] and close[i] > close[i-1] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short on pullback to EMA(20) in downtrend with volume
            elif close[i] >= ema_20[i] and close[i] < close[i-1] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            # Exit on break of EMA(20) in opposite direction
            elif position == 1 and close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position * 0.25 if position != 0 else 0.0
                
        else:  # Transition zone - no trade
            signals[i] = 0.0
            if position != 0:
                # Exit on stop loss only in transition zone
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Kaufman Adaptive Moving Average (KAMA) direction + RSI(14) + choppiness regime filter.
# KAMA adapts