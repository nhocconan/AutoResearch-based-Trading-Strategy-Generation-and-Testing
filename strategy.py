#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# Enter long when price breaks above Donchian(20) upper band, close > 1w EMA(50), volume > 1.5x 20-day average
# Enter short when price breaks below Donchian(20) lower band, close < 1w EMA(50), volume > 1.5x 20-day average
# Exit when price reverses to opposite Donchian band or volume drops below threshold
# Target: 30-100 trades over 4 years (7-25/year) with proper trend filtering to avoid whipsaws

name = "1d_donchian20_1wema_vol_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) bands
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian lower band OR volume drops below threshold
            if close[i] < low_20[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian upper band OR volume drops below threshold
            if close[i] > high_20[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume confirmation
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > ema_50_aligned[i]:
                    # Breakout above upper band with uptrend confirmation
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and close[i] < ema_50_aligned[i]:
                    # Breakout below lower band with downtrend confirmation
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels with 1w ADX trend filter and volume spike confirmation
# Enter long when price touches Camarilla L3 support, ADX > 25 (trending), volume > 2x average
# Enter short when price touches Camarilla H3 resistance, ADX > 25, volume > 2x average
# Exit when price reaches opposite Camarilla level (H3 for long, L3 for short) or ADX < 20
# Camarilla provides precise intraday levels that work well on daily timeframe
# Target: 30-100 trades over 4 years with strict entry conditions to minimize fee drag

name = "1d_camarilla_1wadx_vol_spike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 2:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H2 = close + 1.166 * (high - low)
    # H1 = close + 1.0833 * (high - low)
    # L1 = close - 1.0833 * (high - low)
    # L2 = close - 1.166 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We use H3 for short entries, L3 for long entries
    
    # Shift to use previous day's data for today's levels (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First day uses same day's data
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels based on previous day
    H3 = prev_close + 1.25 * (prev_high - prev_low)
    L3 = prev_close - 1.25 * (prev_high - prev_low)
    
    # 1w ADX for trend filter (ADX > 25 = trending market)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial values
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx[di_plus + di_minus == 0] = 0
        
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        # Set early values to NaN
        adx[:2*period-1] = np.nan
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume spike confirmation: volume > 2x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 since we use previous day's data
        # Skip if required data not available
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price reaches H3 level or ADX drops below 20 (range) or volume drops
            if close[i] >= H3[i] or adx_1w_aligned[i] < 20 or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches L3 level or ADX drops below 20 or volume drops
            if close[i] <= L3[i] or adx_1w_aligned[i] < 20 or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla touch + ADX > 25 + volume spike
            if adx_1w_aligned[i] > 25 and volume[i] > volume_threshold[i]:
                if close[i] <= L3[i]:  # Touched or penetrated L3 support
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= H3[i]:  # Touched or penetrated H3 resistance
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI(2) mean reversion with 1w EMA(200) trend filter and volume confirmation
# Enter long when RSI(2) < 10, price > 1w EMA(200), volume > 1.5x 20-day average
# Enter short when RSI(2) > 90, price < 1w EMA(200), volume > 1.5x 20-day average
# Exit when RSI(2) returns to neutral range (40-60) or opposite extreme reached
# RSI(2) is extremely sensitive and captures short-term overextensions
# Combined with long-term trend filter (EMA200) to avoid counter-trend trades in strong moves
# Target: 30-100 trades over 4 years with strict RSI thresholds to limit frequency

name = "1d_rsi2_1wema200_vol_meanrev_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(2) - extremely short period for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[1] = gain.iloc[1] if hasattr(gain, 'iloc') else gain[1]  # First average
    avg_loss[1] = loss.iloc[1] if hasattr(loss, 'iloc') else loss[1]
    for i in range(2, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain.iloc[i] if hasattr(gain, 'iloc') else gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss.iloc[i] if hasattr(loss, 'iloc') else loss[i]) / 2
    # Handle division by zero
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1w EMA(200) for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start from 2 for RSI(2) calculation
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: RSI returns to neutral (40-60) or reaches overbought (>80)
            if rsi[i] >= 40 or rsi[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI returns to neutral (40-60) or reaches oversold (<20)
            if rsi[i] <= 60 or rsi[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: RSI extreme + trend filter + volume confirmation
            if volume[i] > volume_threshold[i]:
                if rsi[i] < 10 and close[i] > ema_200_aligned[i]:
                    # Extremely oversold but above long-term EMA - bullish mean reversion
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 90 and close[i] < ema_200_aligned[i]:
                    # Extremely overbought but below long-term EMA - bearish mean reversion
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R(14) with 1w ADX trend filter and volume spike confirmation
# Enter long when Williams %R crosses above -20 from below, ADX > 25, volume > 2x average
# Enter short when Williams %R crosses below -80 from above, ADX > 25, volume > 2x average
# Exit when Williams %R reaches opposite extreme (-80 for long, -20 for short) or ADX < 20
# Williams %R is a momentum oscillator that identifies overbought/oversold levels
# Combined with trend filter to capture pullbacks in trending markets
# Target: 30-100 trades over 4 years with strict entry conditions

name = "1d_williamsr_1wadx_vol_spike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 14:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
    
    # 1w ADX for trend filter (ADX > 25 = trending market)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial values
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx[di_plus + di_minus == 0] = 0
        
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        # Set early values to NaN
        adx[:2*period-1] = np.nan
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume spike confirmation: volume > 2x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = np.roll(williams_r, 1)
    prev_williams_r[0] = williams_r[0]  # Initialize first value
    
    for i in range(1, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R reaches -80 or ADX drops below 20 or volume drops
            if williams_r[i] <= -80 or adx_1w_aligned[i] < 20 or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R reaches -20 or ADX drops below 20 or volume drops
            if williams_r[i] >= -20 or adx_1w_aligned[i] < 20 or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R crossover + ADX > 25 + volume spike
            # Cross above -20 from below (bullish)
            if (williams_r[i] > -20 and prev_williams_r[i] <= -20 and 
                adx_1w_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Cross below -80 from above (bearish)
            elif (williams_r[i] < -80 and prev_williams_r[i] >= -80 and 
                  adx_1w_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
        
        prev_williams_r[i] = williams_r[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Stochastic Oscillator(14,3,3) with 1w EMA(50) trend filter and volume confirmation
# Enter long when Stochastic %K crosses above 20 from below, %K > %D, close > 1w EMA(50), volume > 1.5x average
# Enter short when Stochastic %K crosses below 80 from above, %K < %D, close < 1w EMA(50), volume > 1.5x average
# Exit when Stochastic reaches opposite level (80 for long, 20 for short) or trend filter fails
# Stochastic oscillator identifies overbought/oversold levels with smoothing
# Combined with trend filter to capture pullbacks in trending markets
# Target: 30-100 trades over 4 years with strict entry conditions

name = "1d_stochastic_1wema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 14:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Stochastic Oscillator(14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    stoch_k = np.where((highest_high - lowest_low) != 0,
                       ((close - lowest_low) / (highest_high - lowest_low)) * 100, 50)
    
    # %D = 3-period SMA of %K
    stoch_k_series = pd.Series(stoch_k)
    stoch_d = stoch_k_series.rolling(window=3, min_periods=3).mean().values
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_stoch_k = np.roll(stoch_k, 1)
    prev_stoch_k[0] = stoch_k[0]
    prev_stoch_d = np.roll(stoch_d, 1)
    prev_stoch_d[0] = stoch_d[0]
    
    for i in range(1, n):
        # Skip if required data not available
        if (np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Stochastic reaches 80 or trend filter fails or volume drops
            if stoch_k[i] >= 80 or close[i] <= ema_50_aligned[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Stochastic reaches 20 or trend filter fails or volume drops
            if stoch_k[i] <= 20 or close[i] >= ema_50_aligned[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Stochastic crossover + trend filter + volume confirmation
            # Bullish: %K crosses above 20 from below, %K > %D
            if (stoch_k[i] > 20 and prev_stoch_k[i] <= 20 and 
                stoch_k[i] > stoch_d[i] and close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: %K crosses below 80 from above, %K < %D
            elif (stoch_k[i] < 80 and prev_stoch_k[i] >= 80 and 
                  stoch_k[i] < stoch_d[i] and close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
        
        prev_stoch_k[i] = stoch_k[i]
        prev_stoch_d[i] = stoch_d[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Aroon Oscillator(25) with 1w ADX trend filter and volume confirmation
# Enter long when Aroon Oscillator crosses above 0 from below, ADX > 25, volume > 1.5x average
# Enter short when Aroon Oscillator crosses below 0 from above, ADX > 25, volume > 1.5x average
# Exit when Aroon Oscillator reaches extreme (+/-80) or ADX drops below 20
# Aroon Oscillator identifies trend changes by measuring time since highs/lows
# Combined with ADX filter to confirm trend strength
# Target: 30-100 trades over 4 years with strict entry conditions

name = "1d_aroon_1wadx_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Aroon Oscillator(25): Aroon Up - Aroon Down
    # Aroon Up = ((25 - periods since 25-period high) / 25) * 100
    # Aroon Down = ((25 - periods since 25-period low) / 25) * 100
    def calculate_aroon(high, low, period=25):
        aroon_up = np.full_like(high, np.nan, dtype=float)
        aroon_down = np.full_like(low, np.nan, dtype=float)
        
        for i in range(period-1, len(high)):
            # Find highest high in lookback period
            lookback_high = high[i-period+1:i+1]
            periods_since_high = np.argmax(lookback_high[::-1])  # Most recent high
            aroon_up[i] = ((period - 1 - periods_since_high) / (period - 1)) * 100
            
            # Find lowest low in lookback period
            lookback_low = low[i-period+1:i+1]
            periods_since_low = np.argmin(lookback_low[::-1])  # Most recent low
            aroon_down[i] = ((period - 1 - periods_since_low)