#!/usr/bin/env python3
"""
Experiment #079: 4h Primary + 1d HTF — KAMA Adaptive Trend + Fisher Transform + Regime Filter

Hypothesis: Previous 4h strategies failed because they used static indicators (EMA, HMA) that
don't adapt to market conditions. KAMA (Kaufman Adaptive Moving Average) adjusts smoothing
based on market efficiency ratio - smooth in noise, fast in trends. Combined with Ehlers
Fisher Transform for precise reversal entries and Choppiness Index for regime detection,
this should capture both trend and mean-reversion opportunities while adapting to volatility.

Strategy Logic:
1. KAMA(10,2,30) on 4h: Adaptive trend following (ER-based smoothing)
2. EHLERS FISHER TRANSFORM(9): Long when Fisher crosses above -1.5, short when crosses below +1.5
3. CHOPPINESS INDEX(14): CHOP > 55 = range (use Fisher reversals), CHOP < 45 = trend (use KAMA breakout)
4. 1d HMA(21) SLOPE: Major trend bias filter (only trade with 1d trend)
5. ATR(14) stoploss: 2.5x trailing stop
6. Position size: 0.28 discrete (conservative for 4h frequency)

Why this should work:
- KAMA adapts to market conditions (better than static EMA/HMA in chop)
- Fisher Transform catches reversals with less lag than RSI
- Dual regime logic (trend vs mean-revert) based on Choppiness
- 1d HTF prevents counter-trend trades in strong moves
- 4h timeframe targets 20-50 trades/year (fee-efficient)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio calculation
    net_change = np.abs(close_s.diff(er_period))
    sum_changes = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    er = net_change / sum_changes.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Fisher = 0.5 * ln((1 + value) / (1 - value))
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price to -1 to +1 range
        range_val = hh - ll
        if range_val == 0:
            range_val = 1e-10
        
        # Use close for Fisher calculation
        close_price = (high[i] + low[i]) / 2.0
        value = 2.0 * (close_price - ll) / range_val - 1.0
        
        # Clamp to avoid log(0) or log(negative)
        value = np.clip(value, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
        
        # Fisher signal (previous Fisher value for crossover detection)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
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
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    kama_10 = calculate_kama(close, 10, 2, 30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(kama_10[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # HMA slope > 0.3 = bullish bias (prefer longs)
        # HMA slope < -0.3 = bearish bias (prefer shorts)
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range market (use Fisher reversals)
        # CHOP < 45 = trend market (use KAMA breakout)
        # CHOP between = transitional (weaker signals)
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === KAMA TREND SIGNAL ===
        # Price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # KAMA slope (simple momentum)
        kama_slope_bullish = kama_10[i] > kama_10[i-5] if i >= 5 else False
        kama_slope_bearish = kama_10[i] < kama_10[i-5] if i >= 5 else False
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_cross = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_short_cross = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Extreme Fisher levels for stronger signals
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_neutral = 40 <= rsi_14[i] <= 60
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets or neutral 1d trend
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.6
        if trend_1d_neutral:
            current_size = current_size * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if is_range_market:
            # Mean reversion: Fisher reversal from oversold + RSI confirmation
            if fisher_long_cross and rsi_oversold:
                # Only long if 1d trend is bullish or neutral (not strongly bearish)
                if trend_1d_bullish or trend_1d_neutral:
                    new_signal = current_size
            # Also enter on extreme Fisher without cross if very oversold
            elif fisher_oversold and rsi_14[i] < 30:
                if trend_1d_bullish or price_above_1d_hma:
                    new_signal = current_size * 0.8
        
        elif is_trend_market:
            # Trend following: Price above KAMA + KAMA slope up + 1d bullish
            if price_above_kama and kama_slope_bullish:
                if trend_1d_bullish and rsi_neutral:
                    new_signal = current_size
                # Or price above 1d HMA as alternative confirmation
                elif price_above_1d_hma and rsi_14[i] < 55:
                    new_signal = current_size * 0.8
        else:
            # Transitional: Only strong signals with 1d confirmation
            if trend_1d_bullish and price_above_kama and fisher[i] > -1.0:
                new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        if is_range_market:
            # Mean reversion: Fisher reversal from overbought + RSI confirmation
            if fisher_short_cross and rsi_overbought:
                # Only short if 1d trend is bearish or neutral (not strongly bullish)
                if trend_1d_bearish or trend_1d_neutral:
                    new_signal = -current_size
            # Also enter on extreme Fisher without cross if very overbought
            elif fisher_overbought and rsi_14[i] > 70:
                if trend_1d_bearish or price_below_1d_hma:
                    new_signal = -current_size * 0.8
        
        elif is_trend_market:
            # Trend following: Price below KAMA + KAMA slope down + 1d bearish
            if price_below_kama and kama_slope_bearish:
                if trend_1d_bearish and rsi_neutral:
                    new_signal = -current_size
                # Or price below 1d HMA as alternative confirmation
                elif price_below_1d_hma and rsi_14[i] > 45:
                    new_signal = -current_size * 0.8
        else:
            # Transitional: Only strong signals with 1d confirmation
            if trend_1d_bearish and price_below_kama and fisher[i] < 1.0:
                new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~17 days on 4h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and price_above_kama and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and price_below_kama and rsi_14[i] > 55:
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
        # Exit if regime changes strongly against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if market becomes strongly trending bearish on 1d
            if position_side > 0 and is_trend_market and trend_1d_bearish:
                regime_reversal = True
            # Exit short if market becomes strongly trending bullish on 1d
            if position_side < 0 and is_trend_market and trend_1d_bullish:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
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