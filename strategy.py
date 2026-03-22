#!/usr/bin/env python3
"""
Experiment #001: 4h Dual Regime Strategy with 1d HMA Trend Filter

Hypothesis: A dual-regime approach adapting to market conditions will work
across BTC/ETH/SOL in both bull (2021) and bear/range (2022, 2025) markets.

Key design:
1. 1d HMA(21) for major trend direction (call ONCE before loop via mtf_data)
2. 4h Choppiness Index (14) for regime detection:
   - CHOP < 38.2 = trending regime → follow breakouts
   - CHOP > 61.8 = ranging regime → mean revert at extremes
   - Between = neutral → reduced position size
3. 4h Donchian(20) breakout for trend entries
4. 4h RSI(14) extremes for range entries (<35 long, >65 short)
5. ATR(14) for stoploss (2.5x) and volatility filter
6. Discrete sizing: 0.25 base, 0.30 strong trend, 0.20 range

Why this should work:
- Choppiness Index successfully filters whipsaws in 2022 crash (ETH Sharpe +0.923)
- Dual regime captures both trending (2021) and ranging (2022, 2025) markets
- 1d HTF filter prevents counter-trend trades (major failure mode)
- Simple RSI/Donchian ensures trades actually trigger (avoid 0-trade failure)
- 4h TF targets 20-50 trades/year (optimal for fee efficiency)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_1d_hma_chop_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = ranging/choppy market (mean reversion)
    - CHOP < 38.2 = trending market (trend following)
    - Between = neutral/transition
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR over period
    tr_series = pd.Series(tr)
    atr_sum = tr_series.rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    hh = high_s.rolling(window=period, min_periods=period).max().values
    ll = low_s.rolling(window=period, min_periods=period).min().values
    price_range = hh - ll
    
    # Avoid division by zero
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_s / np.where(tr_s == 0, 1e-10, tr_s)
    minus_di = 100 * minus_dm_s / np.where(tr_s == 0, 1e-10, tr_s)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, 1e-10, plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Also calculate 4h HMA for additional trend confirmation
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    RANGE_SIZE = 0.20
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        # === 1D HTF TREND BIAS ===
        htf_bullish = close[i] > hma_1d_21_aligned[i]
        htf_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND ===
        local_bullish = close[i] > hma_4h_21[i]
        local_bearish = close[i] < hma_4h_21[i]
        local_strong_bull = close[i] > hma_4h_50[i]
        local_strong_bear = close[i] < hma_4h_50[i]
        
        # === CHOPPINESS REGIME ===
        regime_trend = chop_14[i] < 38.2  # Trending
        regime_range = chop_14[i] > 61.8  # Ranging
        regime_neutral = not regime_trend and not regime_range
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25
        adx_weak = adx_14[i] <= 25
        
        # === RSI FILTER ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === DONCHIAN BREAKOUT ===
        donchian_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === POSITION SIZING BASED ON REGIME ===
        if regime_trend:
            if htf_bullish and local_bullish and adx_strong:
                current_size = STRONG_SIZE
            elif htf_bullish and local_bullish:
                current_size = BASE_SIZE
            elif htf_bearish and local_bearish and adx_strong:
                current_size = STRONG_SIZE
            elif htf_bearish and local_bearish:
                current_size = BASE_SIZE
            else:
                current_size = WEAK_SIZE
        elif regime_range:
            current_size = RANGE_SIZE
        else:
            current_size = BASE_SIZE * 0.8
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TRENDING REGIME: Follow breakouts with trend
        if regime_trend:
            # LONG: 1d bullish + Donchian breakout + RSI > 50
            if htf_bullish and donchian_long and rsi_bullish:
                new_signal = current_size
            # SHORT: 1d bearish + Donchian breakout + RSI < 50
            elif htf_bearish and donchian_short and rsi_bearish:
                new_signal = -current_size
        
        # RANGING REGIME: Mean revert at extremes
        elif regime_range:
            # LONG: RSI oversold + price near Donchian lower
            if rsi_oversold and close[i] < donchian_lower[i-1] * 1.02 if i > 0 else False:
                new_signal = current_size
            # SHORT: RSI overbought + price near Donchian upper
            elif rsi_overbought and close[i] > donchian_upper[i-1] * 0.98 if i > 0 else False:
                new_signal = -current_size
        
        # NEUTRAL REGIME: Reduced size, require stronger confluence
        elif regime_neutral:
            if htf_bullish and donchian_long and rsi_bullish and adx_strong:
                new_signal = current_size
            elif htf_bearish and donchian_short and rsi_bearish and adx_strong:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~6-7 days on 4h), allow weaker entry
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if htf_bullish and donchian_long:
                new_signal = current_size * 0.7
            elif htf_bearish and donchian_short:
                new_signal = -current_size * 0.7
            # Range entries in neutral regime
            elif regime_range and rsi_oversold:
                new_signal = current_size * 0.7
            elif regime_range and rsi_overbought:
                new_signal = -current_size * 0.7
        
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
            if position_side > 0 and htf_bearish and local_bearish:
                trend_reversal = True
            if position_side < 0 and htf_bullish and local_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (for range trades) ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 30:
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