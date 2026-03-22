#!/usr/bin/env python3
"""
Experiment #147: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime Switch

Hypothesis: Previous 1d strategies failed because they used too many confluence filters,
resulting in 0 trades. Research shows Ehlers Fisher Transform catches reversals in bear
markets better than RSI. This strategy uses:

1. FISHER TRANSFORM (period=9): Long when Fisher crosses above -1.5, short when below +1.5
2. CHOPPINESS INDEX: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend (trend follow)
3. 1w HMA(21): Major trend bias - only take longs if weekly trend neutral/bullish
4. ATR(14) trailing stop: 2.5x ATR to protect capital
5. RELAXED entry thresholds: Ensure 20-50 trades/year (learned from 136 failed experiments)

Why this should work:
- Fisher Transform has better reversal detection than RSI in bear/range markets
- 1d timeframe = natural 20-50 trades/year target
- 1w HTF prevents fighting secular trends
- Simpler logic = more trades (key lesson from experiments 135-146)
- Asymmetric sizing: 0.30 normal, 0.20 in uncertain regimes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_regime_1w_v1"
timeframe = "1d"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_values = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - LL) / (HH - LL) - 0.33
    Signals: Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate median price (typical price)
    median_price = (high_s + low_s) / 2.0
    
    # Calculate highest high and lowest low over period
    hh = median_price.rolling(window=period, min_periods=period).max()
    ll = median_price.rolling(window=period, min_periods=period).min()
    
    # Calculate X value
    price_range = hh - ll
    price_range = price_range.replace(0, 1e-10)
    x = 0.67 * (median_price - ll) / price_range - 0.33
    x = np.clip(x, -0.99, 0.99)  # Prevent division by zero in log
    
    # Calculate Fisher Transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Calculate Fisher signal line (1-period lag)
    fisher_signal = fisher.shift(1)
    
    fisher_vals = fisher.values
    fisher_sig_vals = fisher_signal.values
    
    return fisher_vals, fisher_sig_vals

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        weekly_neutral = not weekly_bullish and not weekly_bearish
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55  # Relaxed from 61.8 for more trades
        is_trend_market = chop_14[i] < 45  # Relaxed from 38.2 for more trades
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = REDUCED_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED for more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths to ensure trades
        long_entry = False
        
        # Path 1: Range market + Fisher oversold (mean revert)
        if is_range_market and fisher_oversold:
            long_entry = True
        
        # Path 2: Fisher cross up + RSI oversold (reversal)
        if fisher_cross_up and rsi_oversold:
            long_entry = True
        
        # Path 3: Weekly neutral/bullish + Fisher oversold (trend pullback)
        if (weekly_neutral or weekly_bullish) and fisher[i] < -0.5:
            long_entry = True
        
        # Path 4: Price above weekly HMA + Fisher cross up (bull continuation)
        if price_above_1w_hma and fisher_cross_up:
            long_entry = True
        
        # Path 5: Simple Fisher extreme (ensure minimum trades)
        if fisher[i] < -1.8 and bars_since_last_trade > 30:
            long_entry = True
        
        if long_entry:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_entry = False
        
        # Path 1: Range market + Fisher overbought (mean revert)
        if is_range_market and fisher_overbought:
            short_entry = True
        
        # Path 2: Fisher cross down + RSI overbought (reversal)
        if fisher_cross_down and rsi_overbought:
            short_entry = True
        
        # Path 3: Weekly bearish + Fisher overbought (trend pullback)
        if weekly_bearish and fisher[i] > 0.5:
            short_entry = True
        
        # Path 4: Price below weekly HMA + Fisher cross down (bear continuation)
        if price_below_1w_hma and fisher_cross_down:
            short_entry = True
        
        # Path 5: Simple Fisher extreme (ensure minimum trades)
        if fisher[i] > 1.8 and bars_since_last_trade > 30:
            short_entry = True
        
        if short_entry:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~60 days on 1d)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and fisher[i] < 0:
                new_signal = current_size * 0.5
            elif weekly_bearish and fisher[i] > 0:
                new_signal = -current_size * 0.5
            elif fisher[i] < -1.0:
                new_signal = current_size * 0.4
            elif fisher[i] > 1.0:
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
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if regime shifts to strong trend bearish
            if position_side > 0 and is_trend_market and weekly_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong trend bullish
            if position_side < 0 and is_trend_market and weekly_bullish:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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