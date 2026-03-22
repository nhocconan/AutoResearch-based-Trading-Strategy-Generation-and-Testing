#!/usr/bin/env python3
"""
Experiment #007: 1d Dual-Regime Strategy with Weekly Trend Filter

Hypothesis: Previous strategies failed because they used ONE approach (pure trend or pure
mean-reversion). This strategy adapts to market regime using ADX:

REGIME 1 - TRENDING (ADX > 25):
  - Follow trend via HMA crossover + Donchian breakout
  - Weekly HMA confirms major direction
  - Enter on pullbacks (RSI 40-60) in trend direction

REGIME 2 - RANGING (ADX < 20):
  - Mean-revert at Bollinger Band extremes
  - RSI oversold/overbought triggers
  - Quick exits at opposite band

Why this should work:
- 2021 bull: ADX high → trend following captures rally
- 2022 crash: ADX spikes → short signals on breakdown
- 2023-2024 range: ADX low → mean-reversion profits from chop
- 2025 bear: ADX moderate → short trend + mean-revert rallies

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-40/year (100-160 over 4 years train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_adx_hma_weekly_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
    close_s = pd.Series(close)
    wma1 = close_s.rolling(window=period//2, min_periods=period//2).mean()
    wma2 = close_s.rolling(window=period, min_periods=period).mean()
    diff = 2 * wma1 - wma2
    hma = diff.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    return atr.values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength meter."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean-reversion entries."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1D indicators
    hma_1d_16 = calculate_hma(close, 16)
    hma_1d_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -40
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (ADX) ===
        trending_regime = adx_14[i] > 25
        ranging_regime = adx_14[i] < 20
        
        # === WEEKLY TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D HMA TREND ===
        hma_bullish = hma_1d_16[i] > hma_1d_48[i]
        hma_bearish = hma_1d_16[i] < hma_1d_48[i]
        
        # === HMA SLOPE ===
        hma_slope_long = hma_1d_16[i] > hma_1d_16[i-10] if i > 10 else False
        hma_slope_short = hma_1d_16[i] < hma_1d_16[i-10] if i > 10 else False
        
        # === TRENDING REGIME LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        if trending_regime:
            # LONG: HMA bullish + weekly bias + pullback entry
            long_score = 0
            if hma_bullish:
                long_score += 2.0
            if weekly_bullish:
                long_score += 1.5
            if hma_slope_long:
                long_score += 1.0
            if rsi_14[i] < 55:  # Pullback entry
                long_score += 1.0
            if close[i] > donchian_lower[i]:  # Above channel bottom
                long_score += 0.5
            
            if long_score >= 4.0:
                new_signal = BASE_SIZE
            
            # SHORT: HMA bearish + weekly bias + pullback entry
            short_score = 0
            if hma_bearish:
                short_score += 2.0
            if weekly_bearish:
                short_score += 1.5
            if hma_slope_short:
                short_score += 1.0
            if rsi_14[i] > 45:  # Pullback entry
                short_score += 1.0
            if close[i] < donchian_upper[i]:  # Below channel top
                short_score += 0.5
            
            if short_score >= 4.0:
                new_signal = -BASE_SIZE
        
        # === RANGING REGIME LOGIC ===
        elif ranging_regime:
            # LONG: RSI oversold + at BB lower
            if rsi_14[i] < 35 and close[i] <= bb_lower[i] * 1.002:
                if weekly_bullish or not weekly_bearish:  # Avoid strong bear
                    new_signal = BASE_SIZE * 0.8
            
            # SHORT: RSI overbought + at BB upper
            if rsi_14[i] > 65 and close[i] >= bb_upper[i] * 0.998:
                if weekly_bearish or not weekly_bullish:  # Avoid strong bull
                    new_signal = -BASE_SIZE * 0.8
        
        # === TRANSITION REGIME (ADX 20-25) ===
        else:
            # Use simpler HMA crossover with weekly filter
            if hma_bullish and weekly_bullish and rsi_14[i] < 50:
                new_signal = BASE_SIZE * 0.7
            elif hma_bearish and weekly_bearish and rsi_14[i] > 50:
                new_signal = -BASE_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~50 days on 1d), allow weaker entry
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if hma_bullish and weekly_bullish:
                new_signal = BASE_SIZE * 0.5
            elif hma_bearish and weekly_bearish:
                new_signal = -BASE_SIZE * 0.5
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # === RANGING REGIME QUICK EXIT ===
        range_exit = False
        if in_position and position_side != 0 and ranging_regime:
            if position_side > 0 and close[i] >= bb_mid[i]:
                range_exit = True  # Take profit at middle
            if position_side < 0 and close[i] <= bb_mid[i]:
                range_exit = True
        
        # Apply exits
        if stoploss_triggered or trend_reversal or range_exit:
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