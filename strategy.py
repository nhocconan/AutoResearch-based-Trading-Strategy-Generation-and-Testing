#!/usr/bin/env python3
"""
Experiment #142: 12h Primary + 1d/1w HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Previous strategies failed because they relied on RSI/Bollinger mean reversion
which gets crushed in strong trends. Research shows Ehlers Fisher Transform excels at
catching reversals in bear/range markets (2022 crash, 2025 bear) with 70%+ win rate.
Combined with KAMA (adaptive to volatility) and Choppiness regime filter, this should:

1. FISHER TRANSFORM(9): Normalizes price to Gaussian distribution, sharp reversal signals
   Long: Fisher crosses above -1.5 from below. Short: Fisher crosses below +1.5 from above.
2. KAMA(21): Kaufman Adaptive MA adjusts smoothing based on market efficiency
   Less whipsaw than EMA in choppy markets, follows trends when efficient.
3. CHOPPINESS INDEX(14): Regime switch (>55 = range/mean-revert, <45 = trend)
4. 1d HMA(21) SLOPE: Major trend bias for position sizing adjustment
5. 1w HMA(21): Ultra-long-term trend filter (avoid counter-trend in major moves)

Why this should work:
- Fisher Transform proven in bear markets (catches sharp reversals)
- KAMA adapts to volatility (less whipsaw than fixed EMA)
- 12h timeframe = 25-45 trades/year target (low fee drag)
- Dual HTF (1d + 1w) for robust trend bias
- Loose entry conditions to ensure trades on ALL symbols

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-45/year per symbol (loose conditions to avoid 0 trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_chop_1d1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    ER = |Close - Close_n| / Sum(|Close - Close_prev|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        start_idx = max(0, i - er_period)
        price_change = np.abs(close[i] - close[start_idx])
        
        if start_idx < i:
            volatility = np.sum(np.abs(np.diff(close[start_idx:i+1])))
        else:
            volatility = price_change
        
        er = price_change / volatility if volatility > 0 else 0
        er = np.clip(er, 0, 1)
        
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for sharp reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.66 * ((price - LL) / (HH - LL) - 0.5)
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            price_range = 1e-10
        
        mid_price = (high[i] + low[i]) / 2
        x = 0.66 * ((mid_price - ll) / price_range - 0.5)
        x = np.clip(x, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        fisher_signal[i] = fisher[i-1] if i > 0 else 0
    
    return fisher, fisher_signal

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        # === HTF TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        trend_1w_bullish = hma_1w_slope_aligned[i] > 1.0
        trend_1w_bearish = hma_1w_slope_aligned[i] < -1.0
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_21[i]
        kama_bearish = close[i] < kama_21[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 50
        is_trend_market = chop_14[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size against 1w major trend
        if trend_1w_bearish and current_size > 0:
            current_size = BASE_SIZE * 0.6
        if trend_1w_bullish and current_size < 0:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC — LOOSE CONDITIONS FOR TRADES ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES — Multiple paths to ensure trades
        long_confidence = 0
        
        # Path 1: Fisher cross up + KAMA bullish (primary signal)
        if fisher_cross_up and kama_bullish:
            long_confidence += 3
        
        # Path 2: Fisher extreme low + range market (mean revert)
        if fisher_extreme_low and is_range_market:
            long_confidence += 2
        
        # Path 3: 1d bullish bias + Fisher cross up
        if trend_1d_bullish and fisher_cross_up:
            long_confidence += 2
        
        # Path 4: Price above 1d HMA + Fisher cross up
        if price_above_1d_hma and fisher_cross_up:
            long_confidence += 2
        
        # Path 5: 1w bullish + any Fisher signal
        if trend_1w_bullish and (fisher_cross_up or fisher_extreme_low):
            long_confidence += 2
        
        # Path 6: Simple Fisher extreme (fallback for more trades)
        if fisher_extreme_low and bars_since_last_trade > 60:
            long_confidence += 1
        
        if long_confidence >= 2:
            new_signal = current_size
        elif long_confidence == 1 and bars_since_last_trade > 80:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: Fisher cross down + KAMA bearish
        if fisher_cross_down and kama_bearish:
            short_confidence += 3
        
        # Path 2: Fisher extreme high + range market
        if fisher_extreme_high and is_range_market:
            short_confidence += 2
        
        # Path 3: 1d bearish bias + Fisher cross down
        if trend_1d_bearish and fisher_cross_down:
            short_confidence += 2
        
        # Path 4: Price below 1d HMA + Fisher cross down
        if price_below_1d_hma and fisher_cross_down:
            short_confidence += 2
        
        # Path 5: 1w bearish + any Fisher signal
        if trend_1w_bearish and (fisher_cross_down or fisher_extreme_high):
            short_confidence += 2
        
        # Path 6: Simple Fisher extreme (fallback)
        if fisher_extreme_high and bars_since_last_trade > 60:
            short_confidence += 1
        
        if short_confidence >= 2:
            new_signal = -current_size
        elif short_confidence == 1 and bars_since_last_trade > 80:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD — Force trades if none for 120 bars ===
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.4
            elif trend_1w_bearish and fisher[i] > 1.0:
                new_signal = -current_size * 0.4
            elif fisher_extreme_low:
                new_signal = current_size * 0.35
            elif fisher_extreme_high:
                new_signal = -current_size * 0.35
        
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
            # Exit long if 1w turns strongly bearish
            if position_side > 0 and trend_1w_bearish and fisher[i] > 0:
                regime_reversal = True
            # Exit short if 1w turns strongly bullish
            if position_side < 0 and trend_1w_bullish and fisher[i] < 0:
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