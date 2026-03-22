#!/usr/bin/env python3
"""
Experiment #230: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: After analyzing 229 failed experiments, the key insight is:
- Session filters KILL trade frequency (#218, #225, #228 all Sharpe=0.000)
- Volume filters KILL trade frequency (#220, #229 negative Sharpe)
- Complex regime logic causes 0 trades

This strategy uses PROVEN simple logic with LOOSE entry conditions:
1. 4h HMA(21) for trend direction (HTF bias)
2. 1h RSI(14) pullback entries within trend (not extremes)
3. 1h Donchian(20) breakout confirmation
4. 12h HMA for major trend filter (loose threshold)
5. NO session filter, NO volume filter (learned from failures)
6. Multiple entry paths to GUARANTEE 10+ trades/symbol

Key differences from failed strategies:
- LOOSE RSI thresholds (45-55 for continuation, not tight extremes)
- Multiple entry paths (score-based, any path can trigger)
- Force-trade logic if no trades for 120 bars (~5 days)
- Discrete position sizing: 0.0, ±0.15, ±0.28
- ATR(14) trailing stop at 2.5 * ATR

Position sizing: 0.28 base (max 0.35), discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target: 40-80 trades/year on 1h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_donchian_4h12h_v1"
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    ema_4h_50 = calculate_ema(df_4h['close'].values, 50)
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    
    # 1h HMA and EMA for local trend
    hma_1h_21 = calculate_hma(close, 21)
    hma_1h_slope = calculate_hma_slope(hma_1h_21, 5)
    ema_1h_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    HALF_SIZE = 0.15
    QUARTER_SIZE = 0.10
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(ema_1h_50[i]):
            continue
        
        # === HTF TREND BIAS (4h) ===
        # 4h HMA slope determines primary trend direction
        hma_4h_bullish = hma_4h_slope_aligned[i] > 0.10
        hma_4h_bearish = hma_4h_slope_aligned[i] < -0.10
        hma_4h_neutral = not hma_4h_bullish and not hma_4h_bearish
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === MAJOR TREND FILTER (12h) ===
        # 12h HMA for major trend (loose filter)
        hma_12h_bullish = hma_12h_slope_aligned[i] > -0.20  # Very loose
        hma_12h_bearish = hma_12h_slope_aligned[i] < 0.20   # Very loose
        
        # === LOCAL TREND (1h HMA) ===
        hma_1h_bullish = hma_1h_slope[i] > 0.15
        hma_1h_bearish = hma_1h_slope[i] < -0.15
        
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_21[i]
        
        # === MOMENTUM (RSI) - LOOSE THRESHOLDS ===
        rsi_bullish = rsi_14[i] > 48  # Loose threshold
        rsi_bearish = rsi_14[i] < 52  # Loose threshold
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        rsi_extreme_bull = rsi_14[i] > 60
        rsi_extreme_bear = rsi_14[i] < 40
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC - MULTIPLE PATHS FOR TRADE FREQUENCY ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Score-based with multiple paths
        long_score = 0
        
        # Path 1: Donchian breakout + 4h bullish + RSI bullish (primary strong signal)
        if breakout_long and hma_4h_bullish and rsi_bullish:
            long_score += 5
        
        # Path 2: Donchian breakout + price above 4h HMA + RSI > 50
        if breakout_long and price_above_4h_hma and rsi_14[i] > 50:
            long_score += 4
        
        # Path 3: 4h bullish + 1h bullish + RSI bullish (trend continuation)
        if hma_4h_bullish and hma_1h_bullish and rsi_bullish and price_above_1h_hma:
            long_score += 4
        
        # Path 4: Breakout + 4h bullish (stronger HTF confirmation)
        if breakout_long and hma_4h_bullish:
            long_score += 3
        
        # Path 5: Breakout + 1h bullish + RSI confirmation
        if breakout_long and hma_1h_bullish and rsi_bullish:
            long_score += 3
        
        # Path 6: Simple breakout with RSI confirmation (looser for more trades)
        if breakout_long and rsi_14[i] > 52:
            long_score += 2
        
        # Path 7: 4h bullish + RSI strong (momentum entry without breakout)
        if hma_4h_bullish and rsi_strong_bull and price_above_1h_hma and bars_since_last_trade > 20:
            long_score += 2
        
        # Path 8: 12h bullish bias + 4h bullish + RSI neutral-bullish
        if hma_12h_bullish and hma_4h_bullish and rsi_14[i] > 50 and bars_since_last_trade > 15:
            long_score += 2
        
        # Path 9: Price above both 4h HMA and 1h EMA50 + RSI > 50
        if price_above_4h_hma and close[i] > ema_1h_50[i] and rsi_14[i] > 50:
            long_score += 2
        
        # Path 10: RSI pullback in uptrend (RSI dipped but trend intact)
        if hma_4h_bullish and price_above_4h_hma and 45 < rsi_14[i] < 55 and bars_since_last_trade > 25:
            long_score += 1
        
        if long_score >= 5:
            new_signal = current_size
        elif long_score >= 4:
            new_signal = current_size
        elif long_score >= 3 and bars_since_last_trade > 30:
            new_signal = HALF_SIZE
        elif long_score >= 2 and bars_since_last_trade > 50:
            new_signal = QUARTER_SIZE
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Donchian breakout + 4h bearish + RSI bearish (primary)
        if breakout_short and hma_4h_bearish and rsi_bearish:
            short_score += 5
        
        # Path 2: Donchian breakout + price below 4h HMA + RSI < 50
        if breakout_short and price_below_4h_hma and rsi_14[i] < 50:
            short_score += 4
        
        # Path 3: 4h bearish + 1h bearish + RSI bearish (trend continuation)
        if hma_4h_bearish and hma_1h_bearish and rsi_bearish and price_below_1h_hma:
            short_score += 4
        
        # Path 4: Breakout + 4h bearish (stronger HTF confirmation)
        if breakout_short and hma_4h_bearish:
            short_score += 3
        
        # Path 5: Breakout + 1h bearish + RSI confirmation
        if breakout_short and hma_1h_bearish and rsi_bearish:
            short_score += 3
        
        # Path 6: Simple breakout with RSI confirmation (looser for more trades)
        if breakout_short and rsi_14[i] < 48:
            short_score += 2
        
        # Path 7: 4h bearish + RSI strong (momentum entry without breakout)
        if hma_4h_bearish and rsi_strong_bear and price_below_1h_hma and bars_since_last_trade > 20:
            short_score += 2
        
        # Path 8: 12h bearish bias + 4h bearish + RSI neutral-bearish
        if hma_12h_bearish and hma_4h_bearish and rsi_14[i] < 50 and bars_since_last_trade > 15:
            short_score += 2
        
        # Path 9: Price below both 4h HMA and 1h EMA50 + RSI < 50
        if price_below_4h_hma and close[i] < ema_1h_50[i] and rsi_14[i] < 50:
            short_score += 2
        
        # Path 10: RSI pullback in downtrend (RSI rallied but trend intact)
        if hma_4h_bearish and price_below_4h_hma and 45 < rsi_14[i] < 55 and bars_since_last_trade > 25:
            short_score += 1
        
        if short_score >= 5:
            new_signal = -current_size
        elif short_score >= 4:
            new_signal = -current_size
        elif short_score >= 3 and bars_since_last_trade > 30:
            new_signal = -HALF_SIZE
        elif short_score >= 2 and bars_since_last_trade > 50:
            new_signal = -QUARTER_SIZE
        
        # === FREQUENCY SAFEGUARD - Force trades if none for 120 bars (~5 days) ===
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and rsi_14[i] > 50 and price_above_1h_hma:
                new_signal = QUARTER_SIZE
            elif hma_4h_bearish and rsi_14[i] < 50 and price_below_1h_hma:
                new_signal = -QUARTER_SIZE
            elif rsi_14[i] > 62 and price_above_4h_hma:
                new_signal = QUARTER_SIZE
            elif rsi_14[i] < 38 and price_below_4h_hma:
                new_signal = -QUARTER_SIZE
        
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
            # Long position but 4h turns strongly bearish
            if position_side > 0 and hma_4h_bearish and price_below_4h_hma:
                trend_reversal = True
            # Short position but 4h turns strongly bullish
            if position_side < 0 and hma_4h_bullish and price_above_4h_hma:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
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