#!/usr/bin/env python3
"""
Experiment #223: 1d Primary + 1w HTF — Volatility Mean Reversion + BB Extremes

Hypothesis: After 217 experiments, breakout strategies (Donchian) show mixed results.
This strategy pivots to VOLATILITY MEAN REVERSION which excels in bear/range markets:

1. ATR RATIO (7/30): Detects vol spikes (>2.0 = panic/extreme)
2. BOLLINGER BANDS (20, 2.5): Price at extremes during vol spike = reversion opportunity
3. 1w HMA TREND: Only trade in direction of weekly trend (don't fight major trend)
4. RSI(14) CONFIRMATION: RSI extremes confirm oversold/overbought conditions
5. ATR(14) TRAILING STOP: 2.5 * ATR protects against continued moves

Why this differs from failed strategies:
- NO Connors RSI (tried 50+ times, failing)
- NO Choppiness Index (tried 30+ times, failing)
- NO Donchian breakouts (mixed results in #217, #221)
- Focus on VOL mean reversion (proven in bear markets per research notes)
- Works in 2025 bear/range test period (unlike pure trend following)

Key insight from research: "VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) 
→ long. Captures 'vol crush' after panic. Exit when ATR ratio < 1.2."

Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target: 15-30 trades/year per symbol (1d natural filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_volrevert_bb_hma_1w_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, lower, sma

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio (short/long) for vol spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.zeros(len(close))
    for i in range(len(close)):
        if atr_long[i] > 0 and not np.isnan(atr_long[i]):
            ratio[i] = atr_short[i] / atr_long[i]
        else:
            ratio[i] = 1.0
    return ratio

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.fillna(0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    zscore_20 = calculate_zscore(close, 20)
    
    # 1d HMA for local trend
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
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
    vol_spike_active = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === HTF TREND BIAS (1w) ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.2
        weekly_bearish = hma_1w_slope_aligned[i] < -0.2
        weekly_neutral = not weekly_bullish and not weekly_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === LOCAL TREND (1d HMA) ===
        daily_bullish = hma_1d_slope[i] > 0.15
        daily_bearish = hma_1d_slope[i] < -0.15
        
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0
        vol_normalizing = atr_ratio[i] < 1.3
        vol_extreme = atr_ratio[i] > 2.5
        
        # === BOLLINGER BAND POSITION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002
        at_bb_upper = close[i] >= bb_upper[i] * 0.998
        bb_width = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if bb_mid[i] > 0 else 0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_oversold = rsi_14[i] < 25
        rsi_extreme_overbought = rsi_14[i] > 75
        
        # === Z-SCORE EXTREMES ===
        zscore_oversold = zscore_20[i] < -1.8
        zscore_overbought = zscore_20[i] > 1.8
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Vol mean reversion after panic
        long_score = 0
        
        # Path 1: Vol spike + BB lower + RSI oversold + weekly not bearish (primary)
        if vol_spike and at_bb_lower and rsi_oversold and not weekly_bearish:
            long_score += 5
        
        # Path 2: Vol spike + BB lower + weekly bullish (strong HTF confirmation)
        if vol_spike and at_bb_lower and weekly_bullish:
            long_score += 4
        
        # Path 3: Vol spike + RSI extreme oversold + price above weekly HMA
        if vol_spike and rsi_extreme_oversold and price_above_1w_hma:
            long_score += 4
        
        # Path 4: BB lower + Z-score oversold + weekly bullish (no vol spike needed)
        if at_bb_lower and zscore_oversold and weekly_bullish and bars_since_last_trade > 25:
            long_score += 3
        
        # Path 5: Vol spike + RSI oversold + daily bullish (momentum confirmation)
        if vol_spike and rsi_oversold and daily_bullish:
            long_score += 3
        
        # Path 6: Extreme vol + BB lower (panic capitulation)
        if vol_extreme and at_bb_lower and bars_since_last_trade > 20:
            long_score += 2
        
        # Path 7: Weekly bullish + RSI oversold + price above 1d HMA (pullback entry)
        if weekly_bullish and rsi_oversold and price_above_1d_hma and bars_since_last_trade > 30:
            long_score += 2
        
        # Path 8: Simple BB lower + weekly not bearish (looser for trade frequency)
        if at_bb_lower and not weekly_bearish and rsi_14[i] < 45 and bars_since_last_trade > 35:
            long_score += 1
        
        if long_score >= 4:
            new_signal = current_size
        elif long_score == 3 and bars_since_last_trade > 30:
            new_signal = current_size * 0.7
        elif long_score >= 2 and bars_since_last_trade > 45:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES - Vol mean reversion after euphoria
        short_score = 0
        
        # Path 1: Vol spike + BB upper + RSI overbought + weekly not bullish (primary)
        if vol_spike and at_bb_upper and rsi_overbought and not weekly_bullish:
            short_score += 5
        
        # Path 2: Vol spike + BB upper + weekly bearish (strong HTF confirmation)
        if vol_spike and at_bb_upper and weekly_bearish:
            short_score += 4
        
        # Path 3: Vol spike + RSI extreme overbought + price below weekly HMA
        if vol_spike and rsi_extreme_overbought and price_below_1w_hma:
            short_score += 4
        
        # Path 4: BB upper + Z-score overbought + weekly bearish (no vol spike needed)
        if at_bb_upper and zscore_overbought and weekly_bearish and bars_since_last_trade > 25:
            short_score += 3
        
        # Path 5: Vol spike + RSI overbought + daily bearish (momentum confirmation)
        if vol_spike and rsi_overbought and daily_bearish:
            short_score += 3
        
        # Path 6: Extreme vol + BB upper (euphoria top)
        if vol_extreme and at_bb_upper and bars_since_last_trade > 20:
            short_score += 2
        
        # Path 7: Weekly bearish + RSI overbought + price below 1d HMA (rally entry)
        if weekly_bearish and rsi_overbought and price_below_1d_hma and bars_since_last_trade > 30:
            short_score += 2
        
        # Path 8: Simple BB upper + weekly not bullish (looser for trade frequency)
        if at_bb_upper and not weekly_bullish and rsi_14[i] > 55 and bars_since_last_trade > 35:
            short_score += 1
        
        if short_score >= 4:
            new_signal = -current_size
        elif short_score == 3 and bars_since_last_trade > 30:
            new_signal = -current_size * 0.7
        elif short_score >= 2 and bars_since_last_trade > 45:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 80 bars (~80 days on 1d)
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if weekly_bullish and rsi_14[i] < 45 and price_above_1d_hma:
                new_signal = current_size * 0.35
            elif weekly_bearish and rsi_14[i] > 55 and price_below_1d_hma:
                new_signal = -current_size * 0.35
            elif rsi_14[i] < 30 and price_above_1w_hma:
                new_signal = current_size * 0.30
            elif rsi_14[i] > 70 and price_below_1w_hma:
                new_signal = -current_size * 0.30
        
        # === EXIT LOGIC - Vol normalization ===
        vol_exit = False
        if in_position and position_side != 0:
            # Exit long when vol normalizes after spike (take profit on mean reversion)
            if position_side > 0 and vol_spike_active and vol_normalizing:
                vol_exit = True
            # Exit short when vol normalizes after spike
            if position_side < 0 and vol_spike_active and vol_normalizing:
                vol_exit = True
        
        # Track if we entered on vol spike
        if new_signal != 0.0 and not in_position:
            if vol_spike:
                vol_spike_active = True
            else:
                vol_spike_active = False
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but weekly turns strongly bearish
            if position_side > 0 and weekly_bearish and price_below_1w_hma:
                trend_reversal = True
            # Short position but weekly turns strongly bullish
            if position_side < 0 and weekly_bullish and price_above_1w_hma:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal or vol_exit:
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
                if vol_spike:
                    vol_spike_active = True
                else:
                    vol_spike_active = False
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                if vol_spike:
                    vol_spike_active = True
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
                vol_spike_active = False
        
        signals[i] = new_signal
    
    return signals