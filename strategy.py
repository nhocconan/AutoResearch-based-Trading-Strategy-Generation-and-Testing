#!/usr/bin/env python3
"""
Experiment #373: 15m Volatility Breakout with 4h HMA Trend + Volume Confirmation

Hypothesis: After 372 failed experiments, the pattern shows:
1. Pure trend-following fails (Supertrend, EMA crossover all negative Sharpe)
2. Pure mean-reversion fails (RSI, Z-score all negative Sharpe)
3. 12h/1d timeframes generate too few trades or miss intraday moves
4. 15m timeframe NOT properly tested with MTF filters

NEW APPROACH for 15m:
1. VOLATILITY BREAKOUT: ATR expansion + price突破 Donchian(20) captures momentum bursts
   - 15m Donchian(20) = 5 hour breakout, catches intraday moves
   - ATR(7)/ATR(30) > 1.5 confirms vol expansion (not false breakout)

2. 4h HMA TREND BIAS: Only trade breakouts in HTF trend direction
   - Long breakout only if price > 4h HMA(21)
   - Short breakout only if price < 4h HMA(21)
   - Filters 60%+ of counter-trend false breakouts

3. VOLUME CONFIRMATION: Breakout must have 1.5x average volume
   - Prevents low-liquidity false breakouts
   - Critical for 15m timeframe (noise reduction)

4. ASYMMETRIC POSITION SIZING: 
   - Long: 0.25 (bull markets have stronger momentum)
   - Short: 0.20 (bear rallies are sharper, need smaller size)
   - Discrete levels minimize fee churn

5. ATR TRAILING STOP (2.0x): Tighter than 12h strategies
   - 15m moves faster, need quicker exits
   - Signal → 0 when price moves 2*ATR against position

6. REGIME FILTER: ADX(14) > 18 for trending, skip when ADX < 18
   - Avoids chop losses in ranging markets
   - Hysteresis: enter at 20, exit at 18

Why 15m should work:
- Faster than 12h/1d (which had too few trades or missed moves)
- Slower than 5m (which had too much noise)
- 4h HMA provides stable bias without lag
- Volume filter critical for 15m noise reduction
- Should generate 50-100 trades/year per symbol (enough for stats)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_vol_breakout_4h_hma_volume_adx_atr_v1"
timeframe = "15m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = strong trend, ADX < 20 = ranging market
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 10:
        return adx
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX = smoothed DX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_donchian_channels(high, low, period=20):
    """
    Calculate Donchian Channels.
    Upper = highest high of last N periods
    Lower = lowest low of last N periods
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio[vol_ratio == np.inf] = np.nan
    return vol_ratio

def calculate_vol_expansion(atr, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility expansion detection."""
    atr_s = pd.Series(atr)
    atr_short = atr_s.ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = atr_s.ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    vol_exp = atr_short / atr_long
    vol_exp[vol_exp == np.inf] = np.nan
    return vol_exp

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    vol_expansion = calculate_vol_expansion(atr, 7, 30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels, asymmetric (Rule 4)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(vol_expansion[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === ADX TREND STRENGTH (with hysteresis) ===
        trending_market = adx[i] > 18
        
        # === VOLATILITY EXPANSION ===
        vol_expand = vol_expansion[i] > 1.5
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long breakout: price breaks above Donchian upper
        long_breakout = close[i] > donchian_upper[i-1] if i > 0 else False
        
        # Short breakout: price breaks below Donchian lower
        short_breakout = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Breakout + 4h bullish bias + ADX + Volume + Vol expansion
        if long_breakout and bull_trend_4h and trending_market and vol_confirm and vol_expand:
            new_signal = SIZE_LONG
        
        # SHORT ENTRY: Breakout + 4h bearish bias + ADX + Volume + Vol expansion
        elif short_breakout and bear_trend_4h and trending_market and vol_confirm and vol_expand:
            new_signal = -SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === ADX DROPS BELOW THRESHOLD (hysteresis) ===
        # Exit if market becomes ranging (ADX < 16)
        if in_position and adx[i] < 16:
            new_signal = 0.0
        
        # === VOLATILITY CONTRACTION EXIT ===
        # Exit if vol expansion collapses (vol_exp < 1.2)
        if in_position and vol_expansion[i] < 1.2:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals