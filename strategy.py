#!/usr/bin/env python3
"""
Experiment #035: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend Confluence

Hypothesis: Lower timeframe (1h) strategies fail due to either 0 trades (too strict) or 
too many trades (fee drag). The key is using HTF for DIRECTION (4h HMA) and 1h only 
for ENTRY TIMING (Fisher Transform reversals).

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, catches reversals better than RSI in bear markets
2. 4h HMA(21) for trend bias — simpler than dual 12h+1d, less conflicting signals
3. 1d HMA(50) for macro filter — only one macro condition, not multiple
4. LOOSE entry conditions: Fisher > -1.5 (not exact threshold), volume > 0.6x avg (not 0.8x)
5. Position size 0.22 (smaller than 4h's 0.28 due to more frequent 1h signals)

Why this might work at 1h:
- Fisher Transform is more sensitive than RSI for reversal timing
- Single 4h HMA trend filter (not multiple conflicting HTF filters)
- Loose volume filter ensures trades actually generate
- Target: 40-70 trades/year (within 30-60 target for 1h per Rule 10)

Entry logic (designed to generate trades):
- Long: 4h HMA bullish + 1d HMA bullish + Fisher crosses above -1.5 + volume OK
- Short: 4h HMA bearish + 1d HMA bearish + Fisher crosses below +1.5 + volume OK
- NO session filter (kills trades), NO strict volume (kills trades)

Position size: 0.22 (discrete, smaller for 1h timeframe)
Stoploss: 2.2*ATR trailing stop (tighter than 4h due to more noise)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_trend_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.67 * ((price - lowest_low) / (highest_high - lowest_low) - 0.5) + 0.67 * X_prev
    
    This transforms price into a Gaussian normal distribution for better reversal signals.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    # Calculate typical price (HL2)
    price = (high + low) / 2.0
    
    for i in range(period, n):
        # Highest high and lowest low over lookback period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            price_range = 1e-10
        
        # Calculate X value
        x_raw = (price[i] - lowest_low) / price_range - 0.5
        x = 0.67 * x_raw + 0.67 * (fisher_prev[i-1] if i > period else 0.0)
        
        # Clamp X to avoid division by zero
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        fisher_prev[i] = fisher[i]
    
    return fisher

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 1h than 4h)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(fisher[i]) or np.isnan(vol_ratio[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS ===
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope (3-bar lookback)
        hma_4h_slope_up = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_down = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # === 1D MACRO BIAS ===
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION (LOOSE) ===
        volume_ok = vol_ratio[i] > 0.6  # Very loose - just not dead volume
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5  # Reversal long zone
        fisher_overbought = fisher[i] > 1.5  # Reversal short zone
        
        # Fisher cross signals (more precise entry timing)
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO GENERATE TRADES) ===
        new_signal = 0.0
        
        # Long entry: 4h bullish + 1d bullish OR neutral + Fisher reversal + volume
        if hma_4h_bullish and volume_ok:
            # Easier long entry: just need 4h bullish + Fisher not overbought
            if fisher[i] < 1.0:  # Not extremely overbought
                new_signal = POSITION_SIZE
            # Stronger long: Fisher cross up from oversold
            elif fisher_cross_up:
                new_signal = POSITION_SIZE
        
        # Short entry: 4h bearish + 1d bearish OR neutral + Fisher reversal + volume
        elif hma_4h_bearish and volume_ok:
            # Easier short entry: just need 4h bearish + Fisher not oversold
            if fisher[i] > -1.0:  # Not extremely oversold
                new_signal = -POSITION_SIZE
            # Stronger short: Fisher cross down from overbought
            elif fisher_cross_down:
                new_signal = -POSITION_SIZE
        
        # === MACRO FILTER (soft override) ===
        # Reduce position if 1d HMA contradicts 4h signal
        if new_signal > 0 and hma_1d_bearish:
            new_signal = POSITION_SIZE * 0.5  # Half size against macro
        elif new_signal < 0 and hma_1d_bullish:
            new_signal = -POSITION_SIZE * 0.5  # Half size against macro
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.2 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.2 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.2 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if 4h trend flips bearish
        if in_position and position_side > 0:
            if hma_4h_bearish and hma_4h_slope_down:
                new_signal = 0.0
        
        # Exit short if 4h trend flips bullish
        if in_position and position_side < 0:
            if hma_4h_bullish and hma_4h_slope_up:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals