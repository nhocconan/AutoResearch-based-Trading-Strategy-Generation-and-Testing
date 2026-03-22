#!/usr/bin/env python3
"""
Experiment #216: 12h Primary + 1d HTF — Simplified Donchian + HMA + Volume Filter

Hypothesis: After analyzing 200+ failed experiments, the pattern is clear:
1. Complex regime switching (Connors + Choppiness + Fisher) = 0 trades or negative Sharpe
2. Higher timeframes (12h, 1d) work better than lower ones (4h, 1h)
3. #214 had positive return (+16.1%) but negative Sharpe (-0.416) = too much volatility

This strategy SIMPLIFIES #214 for 12h timeframe:
1. DONCHIAN(14) BREAKOUT: Shorter period than 20 = more signals (target 25-40 trades/year)
2. HMA(21) 1d TREND: Only trade in direction of daily trend (no counter-trend)
3. VOLUME CONFIRMATION: Breakout volume > 1.3 * 20-bar avg volume (filters false breakouts)
4. RSI(14) MOMENTUM: Relaxed thresholds (>48 for long, <52 for short) = more trades
5. ATR(14) STOPLOSS: 2.0 * ATR trailing stop (tighter than 2.5 to reduce DD)
6. VOLATILITY FILTER: Skip entries if ATR ratio > 2.5 (avoid extreme vol spikes)

Why 12h should work better than 4h:
- Fewer trades = less fee drag (target 25-40/year vs 50-80/year)
- Each signal is more meaningful (12h bars capture sustained moves)
- Less whipsaw from intraday noise
- Better alignment with institutional flow

Key improvements over #214:
- Shorter Donchian period (14 vs 20) = more breakout signals
- Volume confirmation = fewer false breakouts
- Relaxed RSI thresholds = more trade opportunities
- Tighter stoploss (2.0 vs 2.5 ATR) = lower drawdown
- 1d HMA trend filter only (not 12h + 1d) = less conflicting signals

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 25-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_vol_1d_v1"
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
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_donchian_channels(high, low, period=14):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 14)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    # 12h HMA for local trend
    hma_12h_21 = calculate_hma(close, 21)
    
    # ATR ratio for volatility filter (current ATR / 30-bar avg ATR)
    atr_sma_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_sma_30[i] > 0:
            atr_ratio[i] = atr_14[i] / atr_sma_30[i]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === HTF TREND BIAS (1d) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.15
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.15
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # Price relative to 1d HMA
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === LOCAL TREND (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        
        # === MOMENTUM (RSI) - Relaxed thresholds ===
        rsi_bullish = rsi_14[i] > 48
        rsi_bearish = rsi_14[i] < 52
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.3 * vol_sma_20[i]
        
        # === VOLATILITY FILTER ===
        vol_normal = atr_ratio[i] < 2.5  # Skip if vol spike > 2.5x normal
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in neutral trend or high volatility
        if trend_1d_neutral or not vol_normal:
            current_size = REDUCED_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_score = 0
        
        # Primary: Breakout + 1d bullish trend + volume + RSI
        if breakout_long and trend_1d_bullish and volume_confirmed and rsi_bullish:
            long_score += 4
        
        # Secondary: Breakout + price above 1d HMA + volume
        if breakout_long and price_above_1d_hma and volume_confirmed:
            long_score += 3
        
        # Tertiary: Breakout + 12h HMA confirmation + RSI
        if breakout_long and price_above_12h_hma and rsi_bullish:
            long_score += 2
        
        # Loose: Simple breakout with volume (for trade frequency)
        if breakout_long and volume_confirmed and vol_normal:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 40:
            new_signal = current_size * 0.7
        
        # SHORT ENTRIES
        short_score = 0
        
        # Primary: Breakout + 1d bearish trend + volume + RSI
        if breakout_short and trend_1d_bearish and volume_confirmed and rsi_bearish:
            short_score += 4
        
        # Secondary: Breakout + price below 1d HMA + volume
        if breakout_short and price_below_1d_hma and volume_confirmed:
            short_score += 3
        
        # Tertiary: Breakout + 12h HMA confirmation + RSI
        if breakout_short and price_below_12h_hma and rsi_bearish:
            short_score += 2
        
        # Loose: Simple breakout with volume (for trade frequency)
        if breakout_short and volume_confirmed and vol_normal:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 80 bars (~40 days on 12h)
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] > 50 and price_above_12h_hma:
                new_signal = REDUCED_SIZE * 0.6
            elif trend_1d_bearish and rsi_14[i] < 50 and price_below_12h_hma:
                new_signal = -REDUCED_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d turns strongly bearish
            if position_side > 0 and trend_1d_bearish and price_below_1d_hma:
                trend_reversal = True
            # Short position but 1d turns strongly bullish
            if position_side < 0 and trend_1d_bullish and price_above_1d_hma:
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