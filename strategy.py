#!/usr/bin/env python3
"""
Experiment #124: 4h Primary + 12h/1d HTF — Fisher Transform + Connors RSI + Vol Regime

Hypothesis: Previous strategies failed due to over-complexity and strict entry conditions.
Research shows Ehlers Fisher Transform catches reversals in bear rallies (75%+ win rate),
while Connors RSI excels at mean reversion timing. This strategy combines:

1. EHLERS FISHER TRANSFORM (period=9): Long when Fisher crosses above -1.5, short when crosses below +1.5
2. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for entry confirmation
3. VOLATILITY REGIME: ATR(7)/ATR(21) ratio detects expansion vs contraction
4. 12h HMA(21) SLOPE: Major trend bias filter
5. CHOPPINESS INDEX (14): Range (>55) = mean revert, Trend (<45) = pullback entries
6. 1d HMA(50): Ultimate trend filter (avoid counter-trend trades)

Why this should work:
- Fisher Transform is proven for reversal detection in non-trending markets
- Connors RSI has documented 75% win rate for mean reversion
- 4h timeframe targets 20-50 trades/year (manageable fee drag)
- 12h/1d HTF prevents fighting major trends
- Multiple entry paths ensure sufficient trade frequency

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_connors_vol_regime_12h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
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
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price
    Signals: Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate median price
    median = (high_s + low_s) / 2
    
    # Normalize price to -1 to +1 range
    highest = median.rolling(window=period, min_periods=period).max()
    lowest = median.rolling(window=period, min_periods=period).min()
    
    price_range = highest - lowest
    price_range = price_range.replace(0, 1e-10)
    
    x = ((median - lowest) / price_range) * 2 - 1
    x = x.clip(-0.999, 0.999)  # Prevent ln domain errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Calculate 1d HTF indicators
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    hma_1d_slope = calculate_hma_slope(hma_1d_50, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_21 = calculate_atr(high, low, close, 21)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # Volatility regime
    vol_ratio = atr_7 / np.where(atr_21 > 0, atr_21, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(fisher[i]):
            continue
        
        # === 1D TREND BIAS (strongest filter) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_50_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_50_aligned[i]
        
        # === 12H TREND BIAS ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.3
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.3
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === VOLATILITY REGIME ===
        vol_expanding = vol_ratio[i] > 1.3
        vol_contracting = vol_ratio[i] < 0.8
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 15
        crsi_extreme_high = crsi[i] > 85
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_bullish_cross = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_bearish_cross = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        fisher_deep_oversold = fisher[i] < -1.8
        fisher_deep_overbought = fisher[i] > 1.8
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if vol_expanding:
            current_size = REDUCED_SIZE  # Reduce size in high vol
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        long_strength = 0
        
        # Path 1: Fisher bullish cross + CRSI oversold (reversal setup)
        if fisher_bullish_cross and crsi_oversold:
            long_score += 3
            long_strength = current_size
        
        # Path 2: Range market + Fisher deep oversold + CRSI extreme
        if is_range_market and fisher_deep_oversold and crsi_extreme_low:
            long_score += 3
            long_strength = current_size
        
        # Path 3: Vol contraction + BB lower + CRSI oversold (squeeze reversal)
        if vol_contracting and price_below_bb_lower and crsi_oversold:
            long_score += 2
            long_strength = current_size * 0.8
        
        # Path 4: 12h bullish trend + pullback + Fisher cross
        if trend_12h_bullish and crsi[i] < 40 and fisher[i] < -1.0:
            long_score += 2
            long_strength = current_size * 0.8
        
        # Path 5: 1d bullish + price above 1d HMA + CRSI pullback
        if trend_1d_bullish and price_above_1d_hma and crsi[i] < 35:
            long_score += 2
            long_strength = current_size * 0.7
        
        # Path 6: Simple Fisher cross (fallback for trade frequency)
        if fisher_bullish_cross and bars_since_last_trade > 60:
            long_score += 1
            long_strength = REDUCED_SIZE
        
        if long_score >= 2:
            new_signal = long_strength
        elif long_score == 1 and bars_since_last_trade > 80:
            new_signal = REDUCED_SIZE * 0.8
        
        # SHORT ENTRIES
        short_score = 0
        short_strength = 0
        
        # Path 1: Fisher bearish cross + CRSI overbought
        if fisher_bearish_cross and crsi_overbought:
            short_score += 3
            short_strength = current_size
        
        # Path 2: Range market + Fisher deep overbought + CRSI extreme
        if is_range_market and fisher_deep_overbought and crsi_extreme_high:
            short_score += 3
            short_strength = current_size
        
        # Path 3: Vol contraction + BB upper + CRSI overbought
        if vol_contracting and price_above_bb_upper and crsi_overbought:
            short_score += 2
            short_strength = current_size * 0.8
        
        # Path 4: 12h bearish trend + rally + Fisher cross
        if trend_12h_bearish and crsi[i] > 60 and fisher[i] > 1.0:
            short_score += 2
            short_strength = current_size * 0.8
        
        # Path 5: 1d bearish + price below 1d HMA + CRSI rally
        if trend_1d_bearish and price_below_1d_hma and crsi[i] > 65:
            short_score += 2
            short_strength = current_size * 0.7
        
        # Path 6: Simple Fisher cross (fallback)
        if fisher_bearish_cross and bars_since_last_trade > 60:
            short_score += 1
            short_strength = REDUCED_SIZE
        
        if short_score >= 2:
            new_signal = -short_strength
        elif short_score == 1 and bars_since_last_trade > 80:
            new_signal = -REDUCED_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and crsi[i] < 35:
                new_signal = REDUCED_SIZE * 0.5
            elif trend_1d_bearish and crsi[i] > 65:
                new_signal = -REDUCED_SIZE * 0.5
            elif fisher[i] < -1.5:
                new_signal = REDUCED_SIZE * 0.4
            elif fisher[i] > 1.5:
                new_signal = -REDUCED_SIZE * 0.4
        
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
            # Exit long if 1d trend turns strongly bearish
            if position_side > 0 and trend_1d_bearish and hma_1d_slope_aligned[i] < -1.0:
                regime_reversal = True
            # Exit short if 1d trend turns strongly bullish
            if position_side < 0 and trend_1d_bullish and hma_1d_slope_aligned[i] > 1.0:
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