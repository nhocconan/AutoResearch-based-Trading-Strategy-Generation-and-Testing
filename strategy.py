#!/usr/bin/env python3
"""
Experiment #082: 12h Primary + 1d/1w HTF — Fisher Transform + Donchian + Choppiness Regime

Hypothesis: Previous strategies relied heavily on RSI-based mean reversion which fails
in strong trending bear markets (2022, 2025). This strategy uses:

1. EHLERS FISHER TRANSFORM (period=9): Better at catching reversals than RSI in bear markets.
   Long when Fisher crosses above -1.5 from below. Short when crosses below +1.5 from above.
   
2. DONCHIAN CHANNEL (20): Proven breakout indicator that worked on SOL (Sharpe +0.782).
   Confirms trend direction before entry.

3. CHOPPINESS INDEX (14): Regime filter from #076 (worked well). 
   CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (breakout).

4. 1d HMA(21) + 1w HMA(21): Dual HTF trend bias. Only long if both bullish, only short if both bearish.

5. ATR-BASED POSITION SIZING: Reduce size when volatility is high (ATR/Close > threshold).

Why this should beat #076 (Sharpe=0.220):
- Fisher Transform catches reversals earlier than RSI (less lag)
- Donchian adds breakout confirmation (reduces false signals)
- 1w HTF adds major trend filter (prevents counter-trend in macro moves)
- Vol-adjusted sizing reduces drawdown in high-vol periods (2022 crash)

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete, reduced in high vol
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_donchian_chop_1d1w_v1"
timeframe = "12h"
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

def calculate_fisher_transform(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    
    Formula:
    1. Normalize price: (close - lowest_low) / (highest_high - lowest_low)
    2. Scale to -1 to +1: 0.999 * (2 * normalized - 1)
    3. Fisher: 0.5 * ln((1 + scaled) / (1 - scaled))
    4. Signal line: EMA of Fisher (period=3)
    
    Entry: Fisher crosses above Signal from below (long), or below Signal from above (short)
    """
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    highest_high = close_s.rolling(window=period, min_periods=period).max()
    lowest_low = close_s.rolling(window=period, min_periods=period).min()
    
    # Normalize price (0 to 1)
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, 1e-10)  # avoid div by zero
    normalized = (close_s - lowest_low) / price_range
    
    # Scale to -0.999 to +0.999
    scaled = 0.999 * (2 * normalized - 1)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + scaled) / (1 - scaled))
    fisher = fisher.replace([np.inf, -np.inf], np.nan)
    
    # Signal line (EMA of Fisher)
    fisher_s = pd.Series(fisher)
    signal = fisher_s.ewm(span=3, min_periods=3, adjust=False).mean()
    
    return fisher.values, signal.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    # Calculate ATR first
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_values = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)  # avoid div by zero
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Upper = Highest High over period
    Lower = Lowest Low over period
    Middle = (Upper + Lower) / 2
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

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
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(close, 9)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    
    # HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    # Track Fisher crossover state
    prev_fisher_cross_long = False
    prev_fisher_cross_short = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        
        # === 1D + 1W TREND BIAS (MAJOR) ===
        # Both HTF must agree for strong signal
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.2
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.2
        
        # Price vs HTF HMA
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # Strong bias: both 1d and 1w agree
        strong_bullish = trend_1d_bullish and trend_1w_bullish
        strong_bearish = trend_1d_bearish and trend_1w_bearish
        
        # Weak bias: only 1d agrees
        weak_bullish = trend_1d_bullish and price_above_1d_hma
        weak_bearish = trend_1d_bearish and price_below_1d_hma
        
        # === CHOPPINESS REGIME DETECTION ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossover detection
        fisher_cross_long = (fisher[i] > fisher_signal[i]) and (fisher[i-1] <= fisher_signal[i-1])
        fisher_cross_short = (fisher[i] < fisher_signal[i]) and (fisher[i-1] >= fisher_signal[i-1])
        
        # Fisher extreme levels (reversal zones)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === 12H TREND CONFIRMATION ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        vol_ratio = atr_14[i] / close[i]  # ATR as % of price
        if vol_ratio > 0.05:  # High vol (>5% daily ATR equivalent)
            current_size = BASE_SIZE * 0.6
        elif vol_ratio > 0.03:  # Medium vol
            current_size = BASE_SIZE * 0.8
        else:  # Low vol
            current_size = BASE_SIZE
        
        # Reduce size in transitional markets
        if not is_range_market and not is_trend_market:
            current_size = current_size * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if is_range_market:
            # Mean reversion: Fisher oversold + RSI oversold + price above 1w HMA
            if fisher_oversold and rsi_oversold and price_above_1w_hma:
                new_signal = current_size
            # Or Fisher cross long in range
            elif fisher_cross_long and rsi_14[i] < 45:
                new_signal = current_size * 0.7
        elif is_trend_market:
            # Trend following: Donchian breakout + HTF bullish + 12h HMA bullish
            if donchian_breakout_long and (strong_bullish or weak_bullish) and hma_bullish:
                new_signal = current_size
            # Or Fisher cross long with trend confirmation
            elif fisher_cross_long and strong_bullish:
                new_signal = current_size * 0.8
        else:
            # Transitional: require stronger confirmation
            if fisher_cross_long and strong_bullish and hma_bullish:
                new_signal = current_size * 0.6
            elif donchian_breakout_long and price_above_1d_hma and price_above_1w_hma:
                new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        if is_range_market:
            # Mean reversion: Fisher overbought + RSI overbought + price below 1w HMA
            if fisher_overbought and rsi_overbought and price_below_1w_hma:
                new_signal = -current_size
            # Or Fisher cross short in range
            elif fisher_cross_short and rsi_14[i] > 55:
                new_signal = -current_size * 0.7
        elif is_trend_market:
            # Trend following: Donchian breakout + HTF bearish + 12h HMA bearish
            if donchian_breakout_short and (strong_bearish or weak_bearish) and hma_bearish:
                new_signal = -current_size
            # Or Fisher cross short with trend confirmation
            elif fisher_cross_short and strong_bearish:
                new_signal = -current_size * 0.8
        else:
            # Transitional: require stronger confirmation
            if fisher_cross_short and strong_bearish and hma_bearish:
                new_signal = -current_size * 0.6
            elif donchian_breakout_short and price_below_1d_hma and price_below_1w_hma:
                new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~50 days on 12h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if strong_bullish and fisher[i] < -0.5:
                new_signal = current_size * 0.4
            elif strong_bearish and fisher[i] > 0.5:
                new_signal = -current_size * 0.4
        
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if regime changes against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if market becomes strongly trending bearish
            if position_side > 0 and is_trend_market and strong_bearish:
                regime_reversal = True
            # Exit short if market becomes strongly trending bullish
            if position_side < 0 and is_trend_market and strong_bullish:
                regime_reversal = True
        
        # === FISHER REVERSAL EXIT ===
        # Exit when Fisher crosses against position
        fisher_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher_cross_short:
                fisher_reversal = True
            if position_side < 0 and fisher_cross_long:
                fisher_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_reversal or fisher_reversal:
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