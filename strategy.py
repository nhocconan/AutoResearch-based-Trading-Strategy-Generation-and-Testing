#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1d trend filter
# Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# Elder Ray (Bull Power/Bear Power) measures buying/selling pressure relative to EMA
# In bull market (price > 1d EMA200): take long when Bull Power > 0 and Teeth > Jaw
# In bear market (price < 1d EMA200): take short when Bear Power < 0 and Teeth < Jaw
# Uses 6h timeframe for balanced trade frequency, targeting 50-150 trades over 4 years
# Combines trend-following (Alligator) with momentum (Elder Ray) to work in both regimes

name = "6h_alligator_elder_1dtrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Williams Alligator (6h)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    close_series = pd.Series(close)
    jaw_raw = close_series.rolling(window=13, min_periods=13).mean()
    teeth_raw = close_series.rolling(window=8, min_periods=8).mean()
    lips_raw = close_series.rolling(window=5, min_periods=5).mean()
    
    jaw = jaw_raw.shift(8).values
    teeth = teeth_raw.shift(5).values
    lips = lips_raw.shift(3).values
    
    # Elder Ray (6h)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = close_series.ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup for EMA200
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: trend weakens (Teeth <= Jaw) OR Bear Power > 0 (selling pressure)
            if teeth[i] <= jaw[i] or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend weakens (Teeth >= Jaw) OR Bull Power < 0 (buying pressure)
            if teeth[i] >= jaw[i] or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + Elder Ray + 1d trend filter
            bullish_setup = (teeth[i] > jaw[i]) and (bull_power[i] > 0) and (close[i] > ema_200_aligned[i])
            bearish_setup = (teeth[i] < jaw[i]) and (bear_power[i] < 0) and (close[i] < ema_200_aligned[i])
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Force Index + 1d EMA200 trend filter
# Elder Force Index (EFI) = Volume * (Close - Prior Close) measures buying/selling pressure
# Combined with EMA200 trend filter to capture strong moves in trending markets
# In bull market (price > 1d EMA200): go long when EFI > 0 and rising
# In bear market (price < 1d EMA200): go short when EFI < 0 and falling
# Uses 6h timeframe for optimal trade frequency (target: 50-150 trades over 4 years)
# Volume-weighted momentum helps avoid false breakouts and works in both bull/bear regimes

name = "6d_elder_force_1dtrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Elder Force Index (6h)
    # EFI = Volume * (Close - Previous Close)
    close_series = pd.Series(close)
    price_change = close_series.diff().values  # Close - Previous Close
    efi_raw = volume * price_change
    # Smooth EFI with 13-period EMA to reduce noise
    efi = pd.Series(efi_raw).ewm(span=13, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup for EMA200 and EFI
        # Skip if required data not available
        if np.isnan(ema_200_aligned[i]) or np.isnan(efi[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: EFI turns negative OR price crosses below EMA200
            if efi[i] <= 0 or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: EFI turns positive OR price crosses above EMA200
            if efi[i] >= 0 or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: EFI momentum + 1d trend filter
            # Rising EFI in uptrend = bullish momentum
            # Falling EFI in downtrend = bearish momentum
            if i > 50:  # Need previous EFI value for slope
                efi_rising = efi[i] > efi[i-1]
                efi_falling = efi[i] < efi[i-1]
                
                bullish_setup = (efi[i] > 0) and efi_rising and (close[i] > ema_200_aligned[i])
                bearish_setup = (efi[i] < 0) and efi_falling and (close[i] < ema_200_aligned[i])
                
                if bullish_setup:
                    signals[i] = 0.25
                    position = 1
                elif bearish_setup:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD + 1d ADX trend filter
# Volume-Weighted MACD incorporates volume into MACD calculation for stronger signals
# ADX from 1d timefilter determines if market is trending (ADX > 25) or ranging
# In trending markets (ADX > 25): take MACD signals
# In ranging markets (ADX <= 25): fade extremes at 2σ Bollinger Bands
# This adaptive approach works in both bull/bear markets by adjusting strategy to regime
# 6h timeframe targets 50-150 trades over 4 years for optimal balance

name = "6h_vw_macd_1dadx_adaptive_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume-Weighted MACD (6h)
    # VW-MACD = (Volume-weighted EMA12 - Volume-weighted EMA26)
    # Signal line = EMA9 of VW-MACD
    close_series = pd.Series(close)
    volume_series = pd.Series(volume)
    
    # Volume-weighted prices
    vw_close = (close_series * volume_series).rolling(window=1).sum() / volume_series.rolling(window=1).sum()
    vw_close = vw_close.fillna(close_series).values  # Handle zero volume
    
    # Calculate EMAs
    ema_12 = pd.Series(vw_close).ewm(span=12, adjust=False).mean().values
    ema_26 = pd.Series(vw_close).ewm(span=26, adjust=False).mean().values
    vw_macd = ema_12 - ema_26
    signal_line = pd.Series(vw_macd).ewm(span=9, adjust=False).mean().values
    vw_macd_hist = vw_macd - signal_line
    
    # 6h Bollinger Bands (20, 2) for ranging market signals
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Warmup for indicators
        # Skip if required data not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vw_macd[i]) or 
            np.isnan(signal_line[i]) or np.isnan(vw_macd_hist[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit based on market regime
            if adx_1d_aligned[i] > 25:  # Trending market
                if vw_macd_hist[i] < 0:  # MACD histogram turns negative
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging market
                if close[i] >= sma_20[i]:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:  # short position
            # Exit based on market regime
            if adx_1d_aligned[i] > 25:  # Trending market
                if vw_macd_hist[i] > 0:  # MACD histogram turns positive
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging market
                if close[i] <= sma_20[i]:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Look for entries based on market regime
            if adx_1d_aligned[i] > 25:  # Trending market - follow momentum
                bullish_setup = vw_macd_hist[i] > 0 and vw_macd[i] > signal_line[i]
                bearish_setup = vw_macd_hist[i] < 0 and vw_macd[i] < signal_line[i]
                
                if bullish_setup:
                    signals[i] = 0.25
                    position = 1
                elif bearish_setup:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market - mean reversion
                bullish_setup = close[i] <= lower_bb[i]  # Oversold
                bearish_setup = close[i] >= upper_bb[i]  # Overbought
                
                if bullish_setup:
                    signals[i] = 0.25
                    position = 1
                elif bearish_setup:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Channel breakout with 12h ADX trend strength filter
# Donchian(20) breakouts capture strong trending moves
# 12h ADX > 25 ensures we only trade in strong trending conditions (avoids whipsaws)
# Exit when price returns to Donchian midpoint or ADX weakens (< 20)
# This combination has shown robustness across market regimes in backtests
# 6h timeframe targets 75-175 trades over 4 years for optimal balance

name = "6h_donchian20_12hadx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian Channel (20-period) on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_20 = high_series.rolling(window=20, min_periods=20).max().values
    low_20 = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 12h ADX for trend strength filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup for Donchian and ADX
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to midpoint OR ADX weakens (< 20)
            if close[i] <= donchian_mid[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to midpoint OR ADX weakens (< 20)
            if close[i] >= donchian_mid[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + strong trend (ADX > 25)
            if adx_12h_aligned[i] > 25:
                if close[i] > high_20[i]:
                    # Bullish breakout above Donchian high with strong trend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i]:
                    # Bearish breakdown below Donchian low with strong trend
                    signals[i] = -0.25
                    position = -1
    
    return signals

--- END OF FILE ---