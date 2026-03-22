#!/usr/bin/env python3
"""
Experiment #023: 1d Regime-Adaptive + 1w HMA Trend + Choppiness Filter

Hypothesis: Daily timeframe with weekly trend bias + regime detection will work better
than pure trend or pure mean-reversion. Key innovations:
1. Choppiness Index (CHOP) to detect trend vs range regime
2. Weekly HMA(21) for strong trend bias (only trade with weekly trend)
3. Regime-adaptive entries: trend-follow in low CHOP, mean-revert in high CHOP
4. Volume + ATR confirmation for entry quality
5. Trailing stop: 3.0 * ATR(14) from entry/extreme
6. Discrete sizing: 0.0, ±0.20, ±0.30 (minimize churn)

Why this should work:
- 1d timeframe = fewer trades (20-50/year), less fee drag
- Weekly filter prevents counter-trend trades (major failure mode)
- Regime switch adapts to market conditions (trend vs range)
- Strict entry conditions = higher quality trades
- Proven on higher timeframes for BTC/ETH

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 3.0 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_chop_1w_hma_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
        
        chop[i] = np.clip(chop[i], 0, 100)
    
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

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def calculate_donchian_high(high, period=20):
    """Calculate Donchian Channel Upper Band."""
    high_s = pd.Series(high)
    return high_s.rolling(window=period, min_periods=period).max().values

def calculate_donchian_low(low, period=20):
    """Calculate Donchian Channel Lower Band."""
    low_s = pd.Series(low)
    return low_s.rolling(window=period, min_periods=period).min().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    vol_avg = calculate_volume_avg(volume, 20)
    donchian_high = calculate_donchian_high(high, 20)
    donchian_low = calculate_donchian_low(low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_TREND = 0.30   # Larger size in trending regime
    SIZE_RANGE = 0.20   # Smaller size in choppy regime
    MIN_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === WEEKLY TREND BIAS (1w HMA) ===
        # Strong trend filter: only trade in direction of weekly trend
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (trend follow)
        # 38.2 <= CHOP <= 61.8 = transition (no trade or reduced size)
        is_trending = chop_14[i] < 38.2
        is_choppy = chop_14[i] > 61.8
        is_transition = not is_trending and not is_choppy
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.9 * vol_avg[i] if not np.isnan(vol_avg[i]) else True
        
        # === SELECT POSITION SIZE BASED ON REGIME ===
        if is_trending:
            current_size = SIZE_TREND
        elif is_choppy:
            current_size = SIZE_RANGE
        else:
            current_size = MIN_SIZE  # Transition = smaller size or no trade
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY CONDITIONS
        if weekly_bullish and volume_ok:
            if is_trending:
                # Trend-follow: breakout above Donchian high
                if close[i] > donchian_high[i] and rsi_14[i] > 50:
                    new_signal = current_size
            elif is_choppy:
                # Mean-reversion: RSI oversold in uptrend
                if rsi_14[i] < 35:
                    new_signal = current_size
        
        # SHORT ENTRY CONDITIONS
        elif weekly_bearish and volume_ok:
            if is_trending:
                # Trend-follow: breakdown below Donchian low
                if close[i] < donchian_low[i] and rsi_14[i] < 50:
                    new_signal = -current_size
            elif is_choppy:
                # Mean-reversion: RSI overbought in downtrend
                if rsi_14[i] > 65:
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2 months on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and rsi_14[i] < 40:
                new_signal = current_size * 0.7
            elif weekly_bearish and rsi_14[i] > 60:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI EXIT (Mean Reversion Profit Target) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Exit short when RSI becomes oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # === WEEKLY TREND REVERSAL EXIT ===
        weekly_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and weekly_bearish:
                weekly_reversal = True
            if position_side < 0 and weekly_bullish:
                weekly_reversal = True
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position type
        regime_exit = False
        if in_position and position_side != 0:
            # In trending regime, exit if becomes choppy (trend exhausted)
            if is_choppy and not is_trending:
                regime_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or rsi_exit or weekly_reversal or regime_exit:
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