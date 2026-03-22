#!/usr/bin/env python3
"""
Experiment #031: 4h Dual Regime (Choppiness Index) + 1d/1w HMA Trend + ATR Risk

Hypothesis: Market regime detection via Choppiness Index allows adaptive strategy:
1. CHOP > 61.8 (choppy/range): Mean reversion at Bollinger extremes
2. CHOP < 38.2 (trending): Donchian breakout with trend confirmation
3. 1d HMA(21) for intermediate trend bias
4. 1w HMA(21) for major trend bias (call ONCE before loop via mtf_data)
5. ATR(14) stoploss at 2.5x for risk management
6. Discrete sizing: 0.25 base, 0.30 strong confluence (HTF + local agree)

Why this should work:
- Choppiness Index is proven regime filter (ETH Sharpe +0.923 in research)
- Dual logic adapts to market conditions (mean revert in range, trend in breakout)
- 1w HMA adds ultra-high timeframe bias (most strategies only use 1d)
- 4h TF targets 20-50 trades/year (optimal fee efficiency)
- Simpler entry conditions ensure trade generation (learned from 0-trade failures)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_dual_1d_1w_hma_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        tr_sum = 0.0
        highest_high = high[i]
        lowest_low = low[i]
        
        for j in range(i - period + 1, i + 1):
            tr = high[j] - low[j]
            if j > 0:
                tr = max(tr, abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
            highest_high = max(highest_high, high[j])
            lowest_low = min(lowest_low, low[j])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HMA trend
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Local 4h HMA for additional confirmation
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_14[i] > 61.8
        trending_regime = chop_14[i] < 38.2
        neutral_regime = not choppy_regime and not trending_regime
        
        # === 1W HTF TREND BIAS (ultra-high timeframe) ===
        htf_weekly_bullish = close[i] > hma_1w_21_aligned[i]
        htf_weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D HTF TREND BIAS (intermediate timeframe) ===
        htf_daily_bullish = close[i] > hma_1d_21_aligned[i]
        htf_daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND ===
        local_bullish = close[i] > hma_4h_21[i]
        local_bearish = close[i] < hma_4h_21[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === BOLLINGER SIGNALS (for mean reversion) ===
        bb_low = close[i] <= bb_lower[i]
        bb_high = close[i] >= bb_upper[i]
        
        # === DONCHIAN SIGNALS (for trend following) ===
        donchian_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === POSITION SIZING BASED ON TREND CONFLUENCE ===
        # Strong: 1w + 1d + local all agree
        # Medium: 1d + local agree
        # Weak: only 1d or only local
        if htf_weekly_bullish and htf_daily_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_daily_bullish and local_bullish:
            current_size = BASE_SIZE
        elif htf_daily_bullish or local_bullish:
            current_size = WEAK_SIZE
        elif htf_weekly_bearish and htf_daily_bearish and local_bearish:
            current_size = STRONG_SIZE
        elif htf_daily_bearish and local_bearish:
            current_size = BASE_SIZE
        elif htf_daily_bearish or local_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # REGIME 1: CHOPPY (mean reversion)
        if choppy_regime:
            # Long: RSI oversold + price at BB lower + 1d bullish bias
            if rsi_oversold and bb_low and htf_daily_bullish:
                new_signal = current_size
            # Short: RSI overbought + price at BB upper + 1d bearish bias
            elif rsi_overbought and bb_high and htf_daily_bearish:
                new_signal = -current_size
        
        # REGIME 2: TRENDING (breakout following)
        elif trending_regime:
            # Long: Donchian breakout + 1w + 1d bullish + RSI > 50
            if donchian_long and htf_weekly_bullish and htf_daily_bullish and rsi_bullish:
                new_signal = current_size
            # Short: Donchian breakout + 1w + 1d bearish + RSI < 50
            elif donchian_short and htf_weekly_bearish and htf_daily_bearish and rsi_bearish:
                new_signal = -current_size
        
        # REGIME 3: NEUTRAL (use simpler conditions to ensure trades)
        else:
            # Long: 1d bullish + RSI > 50 + local bullish
            if htf_daily_bullish and rsi_bullish and local_bullish:
                new_signal = current_size * 0.8
            # Short: 1d bearish + RSI < 50 + local bearish
            elif htf_daily_bearish and rsi_bearish and local_bearish:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~6-7 days on 4h), allow weaker entry
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if htf_daily_bullish and rsi_bullish:
                new_signal = BASE_SIZE * 0.7
            elif htf_daily_bearish and rsi_bearish:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
            # Exit long if 1d trend turns bearish
            if position_side > 0 and htf_daily_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and htf_daily_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (mean reversion profits) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes very overbought
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            # Exit short when RSI becomes very oversold
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or rsi_exit:
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