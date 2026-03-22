#!/usr/bin/env python3
"""
Experiment #071: 4h Primary + 1d/1w HTF — Volatility Mean Reversion with Trend Filter

Hypothesis: Previous trend-following strategies failed because 2022 whipsaw and 2025 bear/range
markets punish pure trend systems. This strategy uses VOLATILITY MEAN REVERSION:

1. 1d HMA(21) for MAJOR trend bias (only trade WITH trend, not against)
2. 4h Bollinger Band squeeze detection (BW percentile < 20 = expansion coming)
3. 4h RSI(7) extremes for entry timing (not 14 - faster for mean reversion)
4. 4h ATR(14) ratio for vol spike detection (ATR(7)/ATR(30) > 1.8 = vol expansion)
5. Asymmetric entries: Long only when 1d bullish, Short only when 1d bearish
6. Position size: 0.28 discrete (reduced to 0.20 in weak trends)
7. Stoploss: 2.5 * ATR(14) trailing (wider for mean reversion)

Why this should work:
- Mean reversion works better in bear/range markets (2025 test period)
- Bollinger squeeze catches volatility expansions before they happen
- 1d trend filter prevents counter-trend disasters (like 2022 bottom whipsaw)
- RSI(7) is faster than RSI(14) for catching short-term extremes
- Vol ratio filter ensures we enter when volatility is actually expanding
- Fewer but higher quality trades (target 25-40/year on 4h)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 base, 0.20 reduced
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_meanrevert_bb_rsi_1d1w_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma * 100  # Bandwidth as percentage
    return upper, lower, sma, bandwidth

def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate Bollinger Bandwidth percentile over lookback."""
    bb_pct = np.zeros(len(bandwidth))
    for i in range(lookback, len(bandwidth)):
        window = bandwidth[i-lookback:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bb_pct[i] = np.sum(valid <= bandwidth[i]) / len(valid) * 100
        else:
            bb_pct[i] = 50
    return bb_pct

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
        else:
            slope[i] = 0
    return slope

def calculate_vol_ratio(atr_values, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility expansion detection."""
    atr_s = pd.Series(atr_values)
    atr_short = atr_s.ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = atr_s.ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    vol_ratio = atr_short / (atr_long + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_bandwidth, 100)
    
    # Volatility ratio
    vol_ratio = calculate_vol_ratio(atr_14, 7, 30)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_percentile[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR FILTER) ===
        # Only trade WITH the 1d trend direction
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5  # Slope > 0.5%
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5  # Slope < -0.5%
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # Price position vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i] * 1.01
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i] * 0.99
        
        # === BOLLINGER BAND SQUEEZE ===
        # BB Percentile < 25 = bands are compressed (expansion likely)
        bb_squeeze = bb_percentile[i] < 25
        
        # === VOLATILITY EXPANSION ===
        # Vol ratio > 1.5 = volatility is expanding (good for entries)
        vol_expanding = vol_ratio[i] > 1.5
        
        # === RSI EXTREMES (MEAN REVERSION) ===
        # RSI(7) < 25 = oversold (long opportunity in bullish trend)
        # RSI(7) > 75 = overbought (short opportunity in bearish trend)
        rsi_oversold = rsi_7[i] < 25
        rsi_overbought = rsi_7[i] > 75
        
        # RSI(7) < 35 = moderately oversold
        # RSI(7) > 65 = moderately overbought
        rsi_mod_oversold = rsi_7[i] < 35
        rsi_mod_overbought = rsi_7[i] > 65
        
        # === PRICE AT BB BOUNDS ===
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.005
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.995
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in neutral 1d trend
        if trend_1d_neutral:
            current_size = REDUCED_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 1d bullish bias + RSI oversold + price at lower BB or vol expanding
        if trend_1d_bullish or price_above_1d_hma:
            # Strong long: RSI very oversold + at lower BB
            if rsi_oversold and price_at_lower_bb:
                new_signal = current_size
            # Moderate long: RSI moderately oversold + vol expanding + BB squeeze
            elif rsi_mod_oversold and vol_expanding and bb_squeeze:
                new_signal = current_size * 0.8
            # Pullback long: RSI pulling back from oversold + still below SMA
            elif rsi_7[i] > 30 and rsi_7[i-1] < 30 and close[i] < bb_sma[i]:
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES
        # Require: 1d bearish bias + RSI overbought + price at upper BB or vol expanding
        if trend_1d_bearish or price_below_1d_hma:
            # Strong short: RSI very overbought + at upper BB
            if rsi_overbought and price_at_upper_bb:
                new_signal = -current_size
            # Moderate short: RSI moderately overbought + vol expanding + BB squeeze
            elif rsi_mod_overbought and vol_expanding and bb_squeeze:
                new_signal = -current_size * 0.8
            # Pullback short: RSI pulling back from overbought + still above SMA
            elif rsi_7[i] < 70 and rsi_7[i-1] > 70 and close[i] > bb_sma[i]:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~33 days on 4h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_mod_oversold:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and rsi_mod_overbought:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend reverses bearish strongly
            if position_side > 0 and trend_1d_bearish:
                trend_reversal = True
            # Exit short if 1d trend reverses bullish strongly
            if position_side < 0 and trend_1d_bullish:
                trend_reversal = True
        
        # === MEAN REVERSION TARGET EXIT ===
        # Exit when price reaches middle band (BB SMA) after mean reversion entry
        mean_reversion_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and close[i] >= bb_sma[i] * 1.002:
                mean_reversion_exit = True
            if position_side < 0 and close[i] <= bb_sma[i] * 0.998:
                mean_reversion_exit = True
        
        # Apply stoploss, trend reversal, or mean reversion exit
        if stoploss_triggered or trend_reversal or mean_reversion_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals