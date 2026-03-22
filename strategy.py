#!/usr/bin/env python3
"""
Experiment #229: 4h Primary + 1d HTF — Volatility Spike Mean Reversion + HMA Trend

Hypothesis: After 228 experiments, trend-following strategies consistently fail in 
bear/range markets (2022 crash, 2025 bear). Volatility spike mean reversion has 
proven edge for BTC/ETH specifically:

1. VOL SPIKE DETECTION: ATR(7)/ATR(30) > 2.0 signals extreme volatility (panic/euphoria)
2. RSI EXTREMES: RSI(14) < 35 (panic long) or > 65 (euphoria short)
3. 1d HMA TREND FILTER: Only trade with daily trend (never fight HTF)
4. ATR(14) TRAILING STOP: 2.5 * ATR protects against continued volatility
5. DISCRETE SIZING: 0.30 max, reduces to 0.15 on partial exit

Why this should work:
- Vol spikes are transient (mean revert within 3-10 bars)
- Works in BOTH bull and bear markets (panic bottoms, euphoria tops)
- 4h TF = 20-50 trades/year target (optimal for cost model)
- 1d HTF filter prevents counter-trend trades that destroy Sharpe

Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target: 25-50 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_rsi_hma_1d_v1"
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
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volatility spike ratio: ATR(7) / ATR(30)
    vol_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0 and not np.isnan(atr_30[i]):
            vol_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    partial_exit_done = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === HTF TREND BIAS (1d) ===
        # Daily trend determines overall bias
        daily_bullish = hma_1d_slope_aligned[i] > 0.3
        daily_bearish = hma_1d_slope_aligned[i] < -0.3
        daily_neutral = not daily_bullish and not daily_bearish
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 2.0
        vol_extreme = vol_ratio[i] > 2.5
        vol_normalizing = vol_ratio[i] < 1.5  # Vol returning to normal
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_extreme_oversold = rsi_14[i] < 25
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_overbought = rsi_14[i] > 75
        rsi_neutral = 40 < rsi_14[i] < 60
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Vol spike + RSI oversold + Daily trend support
        long_score = 0
        
        # Path 1: Vol spike + extreme oversold + price above daily HMA (strongest)
        if vol_spike and rsi_extreme_oversold and price_above_1d_hma:
            long_score += 5
        
        # Path 2: Vol spike + oversold + daily bullish slope
        if vol_spike and rsi_oversold and daily_bullish:
            long_score += 4
        
        # Path 3: Vol extreme + oversold (any daily trend)
        if vol_extreme and rsi_oversold:
            long_score += 3
        
        # Path 4: Vol spike + price above daily HMA + RSI < 45
        if vol_spike and price_above_1d_hma and rsi_14[i] < 45:
            long_score += 3
        
        # Path 5: Daily bullish + RSI oversold (no vol spike needed if RSI extreme)
        if daily_bullish and rsi_extreme_oversold and bars_since_last_trade > 20:
            long_score += 2
        
        # Path 6: Vol normalizing from extreme + RSI rising from oversold
        if i > 100 and vol_ratio[i-1] > 2.5 and vol_ratio[i] < 2.0 and rsi_14[i] > rsi_14[i-1] and rsi_14[i-1] < 35:
            long_score += 3
        
        if long_score >= 4:
            new_signal = BASE_SIZE
        elif long_score >= 3 and bars_since_last_trade > 15:
            new_signal = BASE_SIZE * 0.7
        elif long_score >= 2 and bars_since_last_trade > 25:
            new_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Vol spike + extreme overbought + price below daily HMA (strongest)
        if vol_spike and rsi_extreme_overbought and price_below_1d_hma:
            short_score += 5
        
        # Path 2: Vol spike + overbought + daily bearish slope
        if vol_spike and rsi_overbought and daily_bearish:
            short_score += 4
        
        # Path 3: Vol extreme + overbought (any daily trend)
        if vol_extreme and rsi_overbought:
            short_score += 3
        
        # Path 4: Vol spike + price below daily HMA + RSI > 55
        if vol_spike and price_below_1d_hma and rsi_14[i] > 55:
            short_score += 3
        
        # Path 5: Daily bearish + RSI overbought (no vol spike needed if RSI extreme)
        if daily_bearish and rsi_extreme_overbought and bars_since_last_trade > 20:
            short_score += 2
        
        # Path 6: Vol normalizing from extreme + RSI falling from overbought
        if i > 100 and vol_ratio[i-1] > 2.5 and vol_ratio[i] < 2.0 and rsi_14[i] < rsi_14[i-1] and rsi_14[i-1] > 65:
            short_score += 3
        
        if short_score >= 4:
            new_signal = -BASE_SIZE
        elif short_score >= 3 and bars_since_last_trade > 15:
            new_signal = -BASE_SIZE * 0.7
        elif short_score >= 2 and bars_since_last_trade > 25:
            new_signal = -BASE_SIZE * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~10 days on 4h)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if daily_bullish and rsi_14[i] < 45 and price_above_1d_hma:
                new_signal = BASE_SIZE * 0.4
            elif daily_bearish and rsi_14[i] > 55 and price_below_1d_hma:
                new_signal = -BASE_SIZE * 0.4
            elif rsi_extreme_oversold and bars_since_last_trade > 80:
                new_signal = BASE_SIZE * 0.3
            elif rsi_extreme_overbought and bars_since_last_trade > 80:
                new_signal = -BASE_SIZE * 0.3
        
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
        
        # === TAKE PROFIT - Partial exit at 2R ===
        take_profit_partial = False
        if in_position and not partial_exit_done and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr_14[i] and new_signal > 0:
                    take_profit_partial = True
                    new_signal = HALF_SIZE  # Reduce to half position
            
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr_14[i] and new_signal < 0:
                    take_profit_partial = True
                    new_signal = -HALF_SIZE  # Reduce to half position
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but daily turns strongly bearish
            if position_side > 0 and daily_bearish and price_below_1d_hma:
                trend_reversal = True
            # Short position but daily turns strongly bullish
            if position_side < 0 and daily_bullish and price_above_1d_hma:
                trend_reversal = True
        
        # === VOLATILITY NORMALIZATION EXIT ===
        # If vol spiked for entry but now normalized, consider exit
        vol_exit = False
        if in_position and position_side != 0 and bars_since_last_trade > 10:
            if vol_normalizing and rsi_neutral:
                vol_exit = True
        
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
            partial_exit_done = False
        
        if take_profit_partial:
            partial_exit_done = True
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                partial_exit_done = False
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                partial_exit_done = False
            elif abs(new_signal) < abs(signals[i-1]) if i > 0 else False:
                # Position reduced (take profit), keep tracking
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
                partial_exit_done = False
        
        signals[i] = new_signal
    
    return signals