#!/usr/bin/env python3
"""
Experiment #009: 4h Dual Regime (Choppiness + Donchian) + 1d HMA Trend

Hypothesis: 4h primary with 1d HTF trend filter will work better than 12h because:
1. 4h captures more intermediate swings while 1d filter prevents counter-trend trades
2. Choppiness Index regime detection (CHOP > 61.8 = range, < 38.2 = trend) allows
   switching between mean reversion (in chop) and breakout (in trend)
3. Donchian(20) breakouts work well on SOL (proven Sharpe +0.782)
4. RSI mean reversion works in ranging markets (proven on ETH)
5. Dual regime approach adapts to market conditions (bull/bear/range)

Why this should beat current best (Sharpe=0.025):
- Regime switching captures both trending and ranging periods
- 1d HMA filter prevents major counter-trend losses (2022 crash protection)
- 4h TF targets 20-50 trades/year (optimal fee efficiency)
- ATR stoploss (2.5x) protects from major drawdowns
- Discrete sizing (0.20-0.30) minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_donchian_1d_hma_v1"
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
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D HTF TREND BIAS ===
        # Simple: price above 1d HMA = bullish bias, below = bearish
        htf_bullish = close[i] > hma_1d_21_aligned[i]
        htf_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND ===
        local_bullish = close[i] > hma_4h_21[i]
        local_bearish = close[i] < hma_4h_21[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = choppy/ranging (mean reversion)
        # CHOP < 38.2 = trending (breakout)
        # 38.2 <= CHOP <= 61.8 = neutral (use ADX)
        is_choppy = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        is_neutral = not is_choppy and not is_trending
        
        # ADX confirmation for trending regime
        adx_strong = adx_14[i] > 25
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === DONCHIAN BREAKOUT ===
        donchian_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_bullish and local_bullish and adx_strong:
            current_size = STRONG_SIZE
        elif htf_bullish and local_bullish:
            current_size = BASE_SIZE
        elif htf_bullish:
            current_size = WEAK_SIZE
        elif htf_bearish and local_bearish and adx_strong:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = BASE_SIZE
        elif htf_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TRENDING REGIME: Donchian breakout with trend confirmation
        if is_trending or (is_neutral and adx_strong):
            # Long: 1d bullish + Donchian breakout + RSI > 50
            if htf_bullish and donchian_long and rsi_bullish:
                new_signal = current_size
            # Short: 1d bearish + Donchian breakout + RSI < 50
            elif htf_bearish and donchian_short and rsi_bearish:
                new_signal = -current_size
        
        # CHOPPY REGIME: RSI mean reversion
        elif is_choppy:
            # Long: RSI oversold + 1d not strongly bearish
            if rsi_oversold and not htf_bearish:
                new_signal = current_size * 0.8  # smaller size in chop
            # Short: RSI overbought + 1d not strongly bullish
            elif rsi_overbought and not htf_bullish:
                new_signal = -current_size * 0.8  # smaller size in chop
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~6-7 days on 4h), allow weaker entry
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if htf_bullish and local_bullish and rsi_bullish:
                new_signal = current_size * 0.7
            elif htf_bearish and local_bearish and rsi_bearish:
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
            # Exit long if 1d trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (take profit in chop regime) ===
        rsi_exit = False
        if in_position and position_side != 0:
            if is_choppy:
                # In chop, exit when RSI reaches opposite extreme
                if position_side > 0 and rsi_14[i] > 65:
                    rsi_exit = True
                if position_side < 0 and rsi_14[i] < 35:
                    rsi_exit = True
            else:
                # In trend, only exit at very extreme RSI
                if position_side > 0 and rsi_14[i] > 80:
                    rsi_exit = True
                if position_side < 0 and rsi_14[i] < 20:
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